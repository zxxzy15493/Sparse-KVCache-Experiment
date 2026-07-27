import time
from numpy import rint
import torch
from termcolor import colored
from .TimeManager import timeManager


class LLM:
    """
    A class representing the LLM (currently support Llama, Qwen and GLM).
    """
    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        fixed_output_length: int = 0, # The fixed output length for decoding. If 0, it will not intervene the output length.
        RECALL: bool = False,
        measure_time: bool = False,
        budget: int = 128
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """
        self.fixed_output_length = fixed_output_length
        self.RECALL = RECALL
        self.measure_time = measure_time
        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map 
        self.budget = budget

        if 'llama' in model_name.lower():
            self.num_layer = 32
        elif 'qwen' in model_name.lower():
            self.num_layer = 28
        elif 'glm' in model_name.lower():
            self.num_layer = 40
        else:
            raise ValueError(f"Model {model_name} is not supported.")

    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')
        layer_prefill_start = time.perf_counter()
        timeManager.layer_prefill_start_event[layer_idx].record()

        timeManager.prefill_preAttn_ffn_start_event[layer_idx].record()

        bsz, seq_len, dim = hidden_states.shape 
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                         layer.input_layernorm_variance_epsilon, 
                                                                         layer.input_layernorm_weight)

        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        torch.cuda.empty_cache()
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        
        self.key_states_shape = key_states.shape
        self.key_states_length = key_states.shape[1]

        timeManager.prefill_preAttn_ffn_end_event[layer_idx].record()

        cluster_start = time.perf_counter()
        timeManager.cluster_start_event[layer_idx].record()
        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        timeManager.cluster_end_event[layer_idx].record()
        cluster_end = time.perf_counter()
        timeManager.cluster_time += (cluster_end - cluster_start)

        torch.cuda.empty_cache()
        prefill_attn_start = time.perf_counter()
        timeManager.prefill_attn_start_event[layer_idx].record()
        temp_attn_out = self.prefill_attention(query_states, key_states, value_states, layer_idx)
        timeManager.prefill_attn_end_event[layer_idx].record()
        prefill_attn_end = time.perf_counter()
        timeManager.prefill_attn_time += (prefill_attn_end - prefill_attn_start)


        sync_start = time.perf_counter()
        timeManager.sync_start_event[layer_idx].record()
        self.kv_cache.sync(layer_idx, start_bdx)
        timeManager.sync_end_event[layer_idx].record()
        sync_end = time.perf_counter()
        timeManager.sync_time += (sync_end - sync_start)


        timeManager.prefill_postAttn_ffn_start_event[layer_idx].record()
        del query_states, key_states, value_states
        torch.cuda.empty_cache()
        
        hidden_states += self.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
        del temp_attn_out
        torch.cuda.empty_cache()

        # post attention
        residual = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                    layer.post_attention_layernorm_variance_epsilon, 
                                                                    layer.post_attention_layernorm_weight)
            hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
        
        hidden_states += residual
        del residual
        torch.cuda.empty_cache()

        timeManager.prefill_postAttn_ffn_end_event[layer_idx].record()

        timeManager.layer_prefill_end_event[layer_idx].record()
        layer_prefill_end = time.perf_counter()
        timeManager.layer_prefill_time += (layer_prefill_end - layer_prefill_start)

        return hidden_states

    def layer_decode(self, layer_idx, hidden_states):

        timeManager.decode_preAttn_ffn_start_event[self.num_layer * timeManager.decode_step + layer_idx].record()
        
        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)

        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        
        # key_states, value_states only use in full attention calculation, in retroinfer attention, they are not used(None).
        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx) 

        timeManager.decode_preAttn_ffn_end_event[self.num_layer * timeManager.decode_step + layer_idx].record()

        decode_attn_start = time.perf_counter()
        timeManager.decode_attn_start_event[self.num_layer * timeManager.decode_step + layer_idx].record()

        # attn_out shape: [bs, seq_len, num_head, head_dim]
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        
        decode_attn_end = time.perf_counter()
        timeManager.decode_attn_end_event[self.num_layer * timeManager.decode_step + layer_idx].record()
        timeManager.decode_attn_time += (decode_attn_end - decode_attn_start)

        timeManager.decode_postAttn_ffn_start_event[self.num_layer * timeManager.decode_step + layer_idx].record()

        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        timeManager.decode_postAttn_ffn_end_event[self.num_layer * timeManager.decode_step + layer_idx].record()

        return hidden_states

    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        # start_bdx mean start batch index

        for start_bdx in range(0, bsz, 1):
            end_bdx = min(bsz, start_bdx + 1)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [1, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
      
        
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        self.key_states_length += 1
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits

    def inference(self, inputs_ids):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        timeManager.prefill_latency = round((prefill_end - prefill_start), 4)

        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))

        print("Start decoding ...")
        decode_start = time.perf_counter()
        self.max_new_length = self.max_new_length if self.fixed_output_length == 0 else self.fixed_output_length
        generate_token = 0

        for _ in range(self.max_new_length-1):
            logits = self.decode_forward(inputs_ids=output_ids)
            output_ids = logits.argmax(dim=-1)
            outputs_ids.append(output_ids)
            generate_token += 1
            timeManager.decode_step += 1
            if 'Llama' in self.model_name and self.fixed_output_length == 0:
                if output_ids[0].item() in [128008, 128001, 128009]:
                    break
            elif 'Qwen' in self.model_name and self.fixed_output_length == 0:
                if output_ids[0].item() in [151645]:
                        break
            
                    
        decode_end = time.perf_counter()
        timeManager.decode_latency = round((decode_end - decode_start), 4)

        print(colored(
            f"Decoding latency: {round((decode_end - decode_start)  / (generate_token), 2)} s/step, "
            f"Throughput: {round(self.batch_size * (generate_token) / (decode_end - decode_start), 2)} tokens/s\n",
            'green'
        ))
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        return outputs_ids

    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None, prefill_method='Full_Flash_Attn'):
        """ LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """

        bs, input_length = inputs_ids.shape
        print(colored(f"\ninput_length is: {input_length}\n", 'cyan'))
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type
        self.prefill_method = prefill_method
        '''
            llm.tokenizer.padding_side = "left"
            valid_start: record the valid start position for each sample in the batch
        '''
        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocate GPU buffers and CPU pin memory ...\n")
        self.init_kv_cache(input_length, valid_start, attn_config)

        outputs = self.inference(inputs_ids)

        return outputs