from transformers import Qwen2ForCausalLM, Qwen2Config
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import torch.nn.functional as F
import gc
import math
from .utils import apply_rotary_pos_emb, layer_norm
import flashinfer
from .attnserver import LSHSparseAttnServer
from termcolor import colored
import time

class LLMLayer:
    def __init__(self, layer_idx, config: Qwen2Config) -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.bq :torch.Tensor = None
        self.bk :torch.Tensor = None
        self.bv :torch.Tensor = None

        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.cos_cache :torch.Tensor = None
        self.sin_cache :torch.Tensor = None

        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_value_slicing = (self.num_key_value_heads * self.head_dim)

        self.intermediate_size = config.intermediate_size
        self.mlp_slice = self.intermediate_size
    
    def init_parameters(self, hf_layer: Qwen2DecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()

        self.bq = hf_layer.self_attn.q_proj.bias.detach() 

        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()

        self.bk = hf_layer.self_attn.k_proj.bias.detach() 

        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()

        self.bv = hf_layer.self_attn.v_proj.bias.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        self.gate_proj :torch.Tensor= hf_layer.mlp.gate_proj.weight.detach()

        self.up_proj :torch.Tensor= hf_layer.mlp.up_proj.weight.detach()

        self.down_proj :torch.Tensor= hf_layer.mlp.down_proj.weight.detach()
        
        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def init_gpu(self, device:str = 'cuda:0'):
        self.bq = self.bq.to(device, non_blocking=True)
        self.bk = self.bk.to(device, non_blocking=True)
        self.bv = self.bv.to(device, non_blocking=True)
        
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

class Qwen2Model:
    def __init__(self, 
        model_name: str,
        K: int = 0,
        L: int = 150,
        batch_size :int = 1,
        max_length :int = 256,
        num_sink_tokens :int = 16,
        num_local_tokens :int = 32,
        generation_buffer :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16,
        RECALL: bool = False,
        fixed_output_length: int = 0,
        ) -> None:
        
        self.RECALL = RECALL
        self.fixed_output_length = fixed_output_length


        self.prefill_latency = 0
        self.decode_latency = 0
        self.Latency = 0
        self.TPOT = 0


        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = Qwen2Config.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length

        
        # self.init_parameters()
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.init_parameters()
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        
        torch.cuda.set_device(0)
        if K > 0:
            self.attention_server = LSHSparseAttnServer(config=self.config, K=K, L=L, batch_size=self.batch_size, 
            num_sink_tokens=num_sink_tokens, num_local_tokens=num_local_tokens, generation_buffer=generation_buffer, 
            max_length=self.max_length, device=self.device,  dtype=self.dtype, use_tensor_cores=True, RECALL=self.RECALL)
        self.k_cache = torch.zeros((max_length, self.num_key_value_heads, self.head_dim), dtype=self.dtype, device=self.device)
        self.v_cache = torch.zeros((max_length, self.num_key_value_heads, self.head_dim), dtype=self.dtype, device=self.device)
        self.chunk_size = 16384
        self.wrt_stream = torch.cuda.Stream()

    def init_parameters(self):
        hf_model = Qwen2ForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
  
        self.inv_freq = hf_model.model.layers[0].self_attn.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = 1.0

        position_ids = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[0]
        self.sin_cache = emb.sin()[0]
        self.cos_cache = self.cos_cache * self.attention_scaling
        self.sin_cache = self.sin_cache * self.attention_scaling
        self.cos_cache = self.cos_cache.to(self.dtype)
        self.sin_cache = self.sin_cache.to(self.dtype)
        self.layers :list[LLMLayer] = []
        
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = LLMLayer(idx, self.config)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
            
        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        input_layernorm_variance_epsilon: float,
        input_layernorm_weight: torch.Tensor,
        wq:torch.Tensor,
        wk:torch.Tensor,
        wv:torch.Tensor,
        bq:torch.Tensor,
        bk:torch.Tensor,
        bv:torch.Tensor,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, input_layernorm_variance_epsilon, input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, wq, bq)
        key_states = F.linear(hidden_states, wk, bk)
        value_states = F.linear(hidden_states, wv, bv)
        
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        return query_states, key_states, value_states
        
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        post_attention_layernorm_variance_epsilon: float,
        post_attention_layernorm_weight: torch.Tensor,
        wo: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):  
        hidden_states = F.linear(attn_output, wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, post_attention_layernorm_variance_epsilon, post_attention_layernorm_weight)
        up = F.linear(hidden_states, up_proj)
        gate = F.linear(hidden_states, gate_proj)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, down_proj)
        hidden_states = residual + hidden_states
        return hidden_states
        
    @torch.inference_mode()
    def layer_compute(self, 
            buffer: LLMLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):
        
        residual = hidden_states
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer.input_layernorm_variance_epsilon,
            buffer.input_layernorm_weight,
            buffer.wq,
            buffer.wk,
            buffer.wv,
            buffer.bq,
            buffer.bk,
            buffer.bv,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        key_states = apply_rotary_pos_emb(key_states, self.cos_cache, self.sin_cache, position_ids)
        query_states = apply_rotary_pos_emb(query_states, self.cos_cache, self.sin_cache, position_ids)
        
        hidden_states = self.attention_server.decode(query_states, key_states, value_states, layer_idx)
        
        hidden_states = self.post_attention_compute(
                        hidden_states, residual,
                        buffer.post_attention_layernorm_variance_epsilon,
                        buffer.post_attention_layernorm_weight,
                        buffer.wo,
                        buffer.gate_proj,
                        buffer.up_proj,
                        buffer.down_proj,
                        )
        
        return hidden_states

    @torch.inference_mode()
    def layer_prefill(self, 
            buffer: LLMLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor,
            request_id: int = 0):

        with torch.cuda.stream(self.wrt_stream):
            residual = hidden_states
            
            for (start, end) in zip(self.chunk_start, self.chunk_end):
                h = layer_norm(hidden_states[:,start:end,:], buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
                bsz, q_len, _ = h.size()
                query_states = F.linear(h, buffer.wq, buffer.bq)
                key_states = F.linear(h, buffer.wk, buffer.bk)
                value_states = F.linear(h, buffer.wv, buffer.bv)
                
                query_states = query_states.view(q_len, self.num_heads, self.head_dim)
                key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
                value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)
                
                key_states = apply_rotary_pos_emb(key_states, self.cos_cache, self.sin_cache, position_ids[start:end])
                query_states = apply_rotary_pos_emb(query_states, self.cos_cache, self.sin_cache, position_ids[start:end])
            
                self.k_cache[start:end].copy_(key_states)
                self.v_cache[start:end].copy_(value_states)
                
                h = flashinfer.prefill.single_prefill_with_kv_cache(
                    q=query_states,
                    k=self.k_cache[:end],
                    v=self.v_cache[:end],
                    causal=True,
                    kv_layout="NHD"
                )
                
                h = h.reshape(bsz, q_len, self.hidden_size)
                h = F.linear(h, buffer.wo)
                residual[:,start:end,:].add_(h)
                
        if layer_idx >= 1:
            self.attention_server.build_table(layer_idx - 1, request_id, self.chunk_end[-1])
        
        self.wrt_stream.synchronize()
        
        with torch.cuda.stream(self.wrt_stream):
            hidden_states = residual
            for (start, end) in zip(self.chunk_start, self.chunk_end):
                h = layer_norm(hidden_states[:,start:end,:], buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
                up = F.linear(h, buffer.up_proj)
                gate = F.linear(h, buffer.gate_proj)
                gate = F.silu(gate)
                h = gate * up
                h = F.linear(h, buffer.down_proj)
                residual[:,start:end,:].add_(h)

        self.attention_server.fill(layer_idx, request_id, self.k_cache, self.v_cache, self.chunk_end[-1])
        if layer_idx == self.num_layers - 1:
            self.attention_server.build_table(layer_idx, request_id, self.chunk_end[-1])
        self.wrt_stream.synchronize()
        return residual
        
    @torch.inference_mode()
    def inference(self,
            k: int,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):
        
        self.attention_server.plan()
        hidden_states = F.embedding(input_ids, self.embed_tokens)
       
        for idx in range(self.num_layers):
                hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
                # if torch.isnan(hidden_states).any():
                #     break
        hidden_states = layer_norm(hidden_states[:,-1:,:], self.norm_variance_epsilon, self.norm_weight)
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits
    
    @torch.inference_mode()
    def prefill(self,
        input_ids: torch.LongTensor,
        request_id : int = 0):
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        self.num_chunk = ((input_ids.shape[1] // self.chunk_size ) if (input_ids.shape[1] % self.chunk_size  > 0) else (input_ids.shape[1] // self.chunk_size  - 1)) + 1
        self.chunk_start = [i * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end = [(i+1) * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end[-1] = input_ids.shape[1]
        self.attention_server.alloc_buffer(input_ids.shape[1])
        
        position_ids = torch.arange(input_ids.shape[1], device=self.device, dtype=torch.int32)
        for idx in range(self.num_layers):
                torch.cuda.synchronize()
                hidden_states = self.layer_prefill(self.layers[idx], idx, hidden_states, position_ids, request_id=request_id)
                
        hidden_states = layer_norm(hidden_states[:,-1:,:], self.norm_variance_epsilon, self.norm_weight)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def generate(self,
        input_ids: torch.LongTensor, 
        max_tokens: int = 128):
        
        generated = input_ids[0].tolist()
        prefix_len = input_ids.shape[1]
        position_ids = torch.arange(prefix_len + max_tokens, device=self.device).unsqueeze(0)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        logits = self.prefill(input_ids=input_ids)

        torch.cuda.synchronize()
        self.prefill_latency = time.perf_counter() - start_time
        print(colored(f"the prefill time is {self.prefill_latency}", 'green')) 

        max_tokens = max_tokens if self.fixed_output_length == 0 else self.fixed_output_length
        decode_start = time.perf_counter()
        generate_token = 0
        for k in range(max_tokens):
            start_time = time.time()
            input_ids = logits.argmax(dim=-1)
            generate_token += 1
            logits = self.inference(k = k, input_ids=input_ids, position_ids=position_ids[:,prefix_len + k:prefix_len + k + 1])
            generated.append(input_ids[0].item())
            if input_ids[0].item() in [151645] and self.fixed_output_length == 0:
                break
            


        torch.cuda.synchronize()
        decode_end = time.perf_counter()
        self.decode_latency = round((decode_end - decode_start), 5)
        self.Latency = self.prefill_latency + self.decode_latency
        self.TPOT = round((decode_end - decode_start)  / (generate_token) * 1000, 5)
        print(colored(f"the decode time is {self.decode_latency}, the total time is {self.Latency}, the TPOT is {self.TPOT}", 'green'))
        self.attention_server.clear()
        self.k_cache.zero_()
        self.v_cache.zero_()
        return generated
    

