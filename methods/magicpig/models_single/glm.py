from transformers import AutoConfig, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import gc
import flashinfer
from .utils import layer_norm
from .attnserver import LSHSparseAttnServer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
import torch.distributed as dist


def apply_glm_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    squeeze_batch = False
    if x.dim() == 3:
        x = x.transpose(0, 1).unsqueeze(0)
        squeeze_batch = True

    if rope_cache.dim() == 3:
        rope_cache = rope_cache.unsqueeze(0)

    bsz, num_heads, seq_len, head_dim = x.size()
    rot_dim = rope_cache.shape[-2] * 2
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    rope_cache = rope_cache[:, :seq_len]

    x_rot = x_rot.reshape(bsz, num_heads, seq_len, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, seq_len, x_rot.size(3), 2)
    x_out = torch.stack(
        [
            x_rot[..., 0] * rope_cache[..., 0] - x_rot[..., 1] * rope_cache[..., 1],
            x_rot[..., 1] * rope_cache[..., 0] + x_rot[..., 0] * rope_cache[..., 1],
        ],
        dim=-1,
    ).flatten(3)
    x_out = torch.cat((x_out, x_pass), dim=-1)

    if squeeze_batch:
        return x_out.squeeze(0).transpose(0, 1).contiguous()
    return x_out


class GLMLayer:
    def __init__(self, layer_idx, config) -> None:
        self.wq: torch.Tensor = None
        self.wk: torch.Tensor = None
        self.wv: torch.Tensor = None
        self.wo: torch.Tensor = None

        self.bq: torch.Tensor = None
        self.bk: torch.Tensor = None
        self.bv: torch.Tensor = None

        self.gate_proj: torch.Tensor = None
        self.up_proj: torch.Tensor = None
        self.down_proj: torch.Tensor = None

        self.input_layernorm_weight: torch.Tensor = None
        self.input_layernorm_variance_epsilon: float = 0.0

        self.post_attention_layernorm_weight: torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon: float = 0.0

        self.layer_idx = layer_idx
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.kv_channels
        self.num_key_value_heads = config.num_key_value_heads
        self.key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.world_size

        self.intermediate_size = config.ffn_hidden_size
        self.mlp_slice = self.intermediate_size // self.world_size

    def init_parameters(self, hf_layer) -> None:
        q_hidden_size = self.num_heads * self.head_dim
        kv_hidden_size = self.num_key_value_heads * self.head_dim

        qkv_weight = hf_layer.self_attention.query_key_value.weight.detach()
        q_weight, k_weight, v_weight = qkv_weight.split(
            [q_hidden_size, kv_hidden_size, kv_hidden_size], dim=0
        )
        self.wq = q_weight.split(q_hidden_size // self.world_size, dim=0)[self.rank]
        self.wk = k_weight.split(self.key_value_slicing, dim=0)[self.rank]
        self.wv = v_weight.split(self.key_value_slicing, dim=0)[self.rank]

        qkv_bias = hf_layer.self_attention.query_key_value.bias
        if qkv_bias is not None:
            qkv_bias = qkv_bias.detach()
            q_bias, k_bias, v_bias = qkv_bias.split(
                [q_hidden_size, kv_hidden_size, kv_hidden_size], dim=0
            )
            self.bq = q_bias.split(q_hidden_size // self.world_size, dim=0)[self.rank]
            self.bk = k_bias.split(self.key_value_slicing, dim=0)[self.rank]
            self.bv = v_bias.split(self.key_value_slicing, dim=0)[self.rank]

        self.wo = hf_layer.self_attention.dense.weight.detach()
        self.wo = self.wo.split(self.hidden_size // self.world_size, dim=1)[self.rank]

        dense_h_to_4h = hf_layer.mlp.dense_h_to_4h.weight.detach()
        gate_proj, up_proj = dense_h_to_4h.split(self.intermediate_size, dim=0)
        self.gate_proj = gate_proj.split(self.mlp_slice, dim=0)[self.rank]
        self.up_proj = up_proj.split(self.mlp_slice, dim=0)[self.rank]

        self.down_proj = hf_layer.mlp.dense_4h_to_h.weight.detach()
        self.down_proj = self.down_proj.split(self.mlp_slice, dim=1)[self.rank]

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.eps

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.eps

    def init_gpu(self, device: str = "cuda:0") -> None:
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj = self.down_proj.to(device, non_blocking=True)

        if self.bq is not None:
            self.bq = self.bq.to(device, non_blocking=True)
            self.bk = self.bk.to(device, non_blocking=True)
            self.bv = self.bv.to(device, non_blocking=True)


class GLMModel:
    def __init__(
        self,
        model_name: str,
        K: int = 0,
        L: int = 150,
        batch_size: int = 1,
        max_length: int = 256,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        RECALL: bool = False,
        fixed_budget: int = 0,
        fixed_output_length: int = 0,
        measure_time: bool = False,
    ) -> None:
        self.RECALL = RECALL
        self.fixed_budget = fixed_budget
        self.fixed_output_length = fixed_output_length
        self.measure_time = measure_time

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = model_name
        self.max_length = max_length

        if not hasattr(self.config, "num_hidden_layers"):
            self.config.num_hidden_layers = getattr(self.config, "num_layers")
        if not hasattr(self.config, "num_key_value_heads"):
            if getattr(self.config, "multi_query_attention", False):
                self.config.num_key_value_heads = self.config.multi_query_group_num
            else:
                self.config.num_key_value_heads = self.config.num_attention_heads
        if not hasattr(self.config, "max_position_embeddings"):
            self.config.max_position_embeddings = getattr(self.config, "seq_length", max_length)

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads // self.world_size
        self.head_dim = self.config.kv_channels
        self.num_key_value_heads = self.config.num_key_value_heads // self.world_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings

        self.init_parameters()

        if self.RECALL:
            self.past_key_states = [None for _ in range(self.num_layers)]
        else:
            self.past_key_states = None

        torch.cuda.set_device(self.rank)
        if K > 0 and self.world_size == 1:
            self.attention_server = LSHSparseAttnServer(
                config=self.config,
                K=K,
                L=L,
                batch_size=self.batch_size,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                use_tensor_cores=True,
                RECALL=self.RECALL,
                fixed_budget=self.fixed_budget,
            )

        self.k_cache = torch.zeros(
            (max_length, self.num_key_value_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_cache = torch.zeros(
            (max_length, self.num_key_value_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.chunk_size = 16384
        self.wrt_stream = torch.cuda.Stream()
        eos_token_id = getattr(self.config, "eos_token_id", [])
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_ids = set(eos_token_id)

    def init_parameters(self) -> None:
        # hf_model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     torch_dtype=self.dtype,
        #     trust_remote_code=True,
        # )

        common_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "cuda",
            "trust_remote_code": True,
        }
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        auto_map = getattr(config, "auto_map", {}) or {}
        class_ref = auto_map.get("AutoModelForCausalLM")
        

        model_class = get_class_from_dynamic_module(class_ref, self.model_name)
        hf_model = model_class.from_pretrained(self.model_name, **common_kwargs)
        self.embed_tokens = hf_model.transformer.embedding.word_embeddings.weight.detach().to(self.device)
        self.lm_head = hf_model.transformer.output_layer.weight.detach().to(self.device)

        final_norm = hf_model.transformer.encoder.final_layernorm
        self.norm_weight = final_norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = final_norm.eps

        self.rope_cache = hf_model.transformer.rotary_pos_emb(self.max_length).detach().to(self.device)
        self.rope_cache = self.rope_cache.to(self.dtype)

        self.layers: list[GLMLayer] = []
        for idx, hf_layer in enumerate(hf_model.transformer.encoder.layers):
            layer = GLMLayer(idx, self.config)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.transformer.encoder.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def _reset_past_key_states(self) -> None:
        if self.RECALL:
            self.past_key_states = [None for _ in range(self.num_layers)]

    def _append_past_key_states(self, layer_idx: int, key_states: torch.Tensor) -> None:
        if not self.RECALL:
            return

        if key_states.dim() == 3:
            key_states = key_states.transpose(0, 1).unsqueeze(0)
        elif key_states.dim() != 4:
            raise ValueError(f"Unsupported key_states shape: {tuple(key_states.shape)}")

        key_states = key_states.contiguous()
        if self.past_key_states[layer_idx] is None:
            self.past_key_states[layer_idx] = key_states
        else:
            self.past_key_states[layer_idx] = torch.cat(
                [self.past_key_states[layer_idx], key_states],
                dim=2,
            ).contiguous()

    def _get_rope_cache(self, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.dim() == 1:
            return self.rope_cache[position_ids.long()].unsqueeze(0)
        return self.rope_cache[position_ids.long()]

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        input_layernorm_variance_epsilon: float,
        input_layernorm_weight: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
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
        dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = layer_norm(
            hidden_states,
            post_attention_layernorm_variance_epsilon,
            post_attention_layernorm_weight,
        )
        gate = F.linear(hidden_states, gate_proj)
        up = F.linear(hidden_states, up_proj)
        hidden_states = F.silu(gate) * up
        hidden_states = F.linear(hidden_states, down_proj)
        dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
        hidden_states = residual + hidden_states
        return hidden_states

    @torch.inference_mode()
    def layer_compute(
        self,
        buffer: GLMLayer,
        layer_idx: int,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
    ):
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
            self.head_dim,
        )

        rope_cache = self._get_rope_cache(position_ids)
        query_states = apply_glm_rotary_pos_emb(query_states, rope_cache)
        key_states = apply_glm_rotary_pos_emb(key_states, rope_cache)
        self._append_past_key_states(layer_idx, key_states)

        if self.RECALL:
            hidden_states = self.attention_server.decode(
                query_states, key_states, value_states, layer_idx, self.past_key_states[layer_idx]
            )
        else:
            hidden_states = self.attention_server.decode(query_states, key_states, value_states, layer_idx)

        hidden_states = self.post_attention_compute(
            hidden_states,
            residual,
            buffer.post_attention_layernorm_variance_epsilon,
            buffer.post_attention_layernorm_weight,
            buffer.wo,
            buffer.gate_proj,
            buffer.up_proj,
            buffer.down_proj,
        )
        return hidden_states

    @torch.inference_mode()
    def layer_prefill(
        self,
        buffer: GLMLayer,
        layer_idx: int,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        request_id: int = 0,
    ):
        with torch.cuda.stream(self.wrt_stream):
            residual = hidden_states

            for (start, end) in zip(self.chunk_start, self.chunk_end):
                h = layer_norm(
                    hidden_states[:, start:end, :],
                    buffer.input_layernorm_variance_epsilon,
                    buffer.input_layernorm_weight,
                )
                bsz, q_len, _ = h.size()
                query_states = F.linear(h, buffer.wq, buffer.bq)
                key_states = F.linear(h, buffer.wk, buffer.bk)
                value_states = F.linear(h, buffer.wv, buffer.bv)

                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                rope_cache = self._get_rope_cache(position_ids[start:end])
                query_states = apply_glm_rotary_pos_emb(query_states, rope_cache)
                key_states = apply_glm_rotary_pos_emb(key_states, rope_cache)
                self._append_past_key_states(layer_idx, key_states)

                query_states = query_states.squeeze(0).transpose(0, 1).contiguous()
                key_states = key_states.squeeze(0).transpose(0, 1).contiguous()
                value_states = value_states.squeeze(0).transpose(0, 1).contiguous()

                self.k_cache[start:end].copy_(key_states)
                self.v_cache[start:end].copy_(value_states)

                h = flashinfer.prefill.single_prefill_with_kv_cache(
                    q=query_states,
                    k=self.k_cache[:end],
                    v=self.v_cache[:end],
                    causal=True,
                    kv_layout="NHD",
                )

                h = h.reshape(bsz, q_len, self.hidden_size // self.world_size)
                h = F.linear(h, buffer.wo)
                dist.all_reduce(h, dist.ReduceOp.SUM)
                residual[:, start:end, :].add_(h)

        if layer_idx >= 1:
            self.attention_server.build_table(layer_idx - 1, request_id, self.chunk_end[-1])

        self.wrt_stream.synchronize()
        with torch.cuda.stream(self.wrt_stream):
            hidden_states = residual
            for (start, end) in zip(self.chunk_start, self.chunk_end):
                h = layer_norm(
                    hidden_states[:, start:end, :],
                    buffer.post_attention_layernorm_variance_epsilon,
                    buffer.post_attention_layernorm_weight,
                )
                gate = F.linear(h, buffer.gate_proj)
                up = F.linear(h, buffer.up_proj)
                h = F.silu(gate) * up
                h = F.linear(h, buffer.down_proj)
                dist.all_reduce(h, dist.ReduceOp.SUM)
                residual[:, start:end, :].add_(h)

        self.attention_server.fill(layer_idx, request_id, self.k_cache, self.v_cache, self.chunk_end[-1])
        if layer_idx == self.num_layers - 1:
            self.attention_server.build_table(layer_idx, request_id, self.chunk_end[-1])

        self.wrt_stream.synchronize()
        return residual

    @torch.inference_mode()
    def inference(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
    ):
        self.attention_server.plan()
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)

        hidden_states = layer_norm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.LongTensor,
        request_id: int = 0,
    ):
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        self.kv_offload_cpu_time = 0.0
        self.attention_server.kv_offload_cpu_time = 0.0
        self._reset_past_key_states()

        self.num_chunk = (
            (input_ids.shape[1] // self.chunk_size)
            if (input_ids.shape[1] % self.chunk_size > 0)
            else (input_ids.shape[1] // self.chunk_size - 1)
        ) + 1
        self.chunk_start = [i * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end = [(i + 1) * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end[-1] = input_ids.shape[1]

        self.attention_server.alloc_buffer(input_ids.shape[1])

        position_ids = torch.arange(input_ids.shape[1], device=self.device, dtype=torch.long)
        for idx in range(self.num_layers):
            torch.cuda.synchronize()
            hidden_states = self.layer_prefill(
                self.layers[idx], idx, hidden_states, position_ids, request_id=request_id
            )

        self.kv_offload_cpu_time = self.attention_server.kv_offload_cpu_time
        hidden_states = layer_norm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_tokens: int = 128,
    ):
        generated = input_ids[0].tolist()
        prefix_len = input_ids.shape[1]
        position_ids = torch.arange(prefix_len + max_tokens, device=self.device).unsqueeze(0)

        logits = self.prefill(input_ids=input_ids)

        max_tokens = max_tokens if self.fixed_output_length == 0 else self.fixed_output_length
        for k in range(max_tokens):
            input_ids = logits.argmax(dim=-1)
            dist.broadcast(input_ids, 0)
            generated.append(input_ids[0].item())
            logits = self.inference(
                input_ids=input_ids,
                position_ids=position_ids[:, prefix_len + k : prefix_len + k + 1],
            )
            if input_ids[0].item() in self.eos_token_ids and self.fixed_output_length == 0:
                break

        self.attention_server.clear()
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._reset_past_key_states()
        return generated

    def clear(self):
        self.attention_server.clear()
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offload_cpu_time = 0.0
        self.attention_server.kv_offload_cpu_time = 0.0
        self._reset_past_key_states()
