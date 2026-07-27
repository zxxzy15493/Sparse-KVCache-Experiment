import gc
import re
import os
import json
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .LLM import LLM
from cache_hub import flash_attn_cache, retroinfer_cache
from attn_hub import prefill_full_flash_attn, decode_full_flash_attn, retroinfer_prefill_attn, retroinfer_decode_attn, prefill_minfer
from .minfer_patterns import llama_31_8b_best_patterns, llama_3_8b_best_patterns, glm_4_9b_chat_1m


class GlmLayer:
    """
    A class representing the ChatGLM layer.
    """

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device

    def init_layer(self, hf_glm_layer):
        attn = hf_glm_layer.self_attention
        mlp = hf_glm_layer.mlp

        self.wqkv = attn.query_key_value.weight.detach().to(self.device, non_blocking=True)
        self.bqkv = attn.query_key_value.bias.detach().to(self.device, non_blocking=True) if attn.query_key_value.bias is not None else None

        self.wo = attn.dense.weight.detach().to(self.device, non_blocking=True)
        self.bo = attn.dense.bias.detach().to(self.device, non_blocking=True) if attn.dense.bias is not None else None

        self.gate_up_proj = mlp.dense_h_to_4h.weight.detach().to(self.device, non_blocking=True)
        self.gate_up_proj_bias = mlp.dense_h_to_4h.bias.detach().to(self.device, non_blocking=True) if mlp.dense_h_to_4h.bias is not None else None

        self.down_proj = mlp.dense_4h_to_h.weight.detach().to(self.device, non_blocking=True)
        self.down_proj_bias = mlp.dense_4h_to_h.bias.detach().to(self.device, non_blocking=True) if mlp.dense_4h_to_h.bias is not None else None

        self.input_layernorm_weight = hf_glm_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_glm_layer.input_layernorm.eps

        self.post_attention_layernorm_weight = hf_glm_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_glm_layer.post_attention_layernorm.eps


class GlmModel(LLM):
    """
    A class representing the GLM model.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        RECALL: bool = False,
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map, RECALL)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if getattr(self.config, "model_type", None) != "chatglm":
            raise ValueError(f"Unsupported GLM config type: {self.config.__class__.__name__}")

        self.num_layers = self.config.num_layers
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.config.kv_channels if self.config.kv_channels is not None else self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.multi_query_group_num if self.config.multi_query_attention else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.seq_length
        self.vocab_size = self.config.padded_vocab_size
        self.use_rmsnorm = self.config.rmsnorm
        eos_token_id = self.config.eos_token_id
        self.eos_tokens = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]


        self.best_patterns = glm_4_9b_chat_1m
        self.init_model()

    def _set_rotary_cache(self, rotary_pos_emb, device):
        rotary_cache = rotary_pos_emb(self.max_length)
        return rotary_cache.to(device=device, dtype=self.dtype, non_blocking=True)

    def init_model(self):
        hf_glm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )

        self.num_gpus = torch.cuda.device_count() if self.device_map == 'auto' else 1
        if self.device_map == 'auto' and self.num_gpus == 1:
            self.device_map = 'cuda:0'

        if self.device_map != "auto":
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): self.device_map})

            self.embed_tokens = hf_glm.transformer.embedding.word_embeddings.weight.detach().to(self.device_map, non_blocking=True)
            self.lm_head = hf_glm.transformer.output_layer.weight.detach().to(self.device_map, non_blocking=True)

            self.norm_weight = hf_glm.transformer.encoder.final_layernorm.weight.detach().to(self.device_map, non_blocking=True)
            self.norm_variance_epsilon = hf_glm.transformer.encoder.final_layernorm.eps

            self.position_ids = torch.arange(0, self.max_length).to(self.device_map, non_blocking=True)
            self.rotary_cache = self._set_rotary_cache(hf_glm.transformer.rotary_pos_emb, self.device_map)

            self.layers = []
            for idx, hf_glm_layer in enumerate(hf_glm.transformer.encoder.layers):
                glm_layer = GlmLayer(idx, device=self.device_map)
                glm_layer.init_layer(hf_glm_layer)
                self.layers.append(glm_layer)
                hf_glm.transformer.encoder.layers[idx] = None
        else:
            self.gpu_ids = list(range(self.num_gpus))
            self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): f'cuda:{ldx // self.layer_interval}'})

            first_device = f'cuda:{self.gpu_ids[0]}'
            self.embed_tokens = hf_glm.transformer.embedding.word_embeddings.weight.detach().to(first_device, non_blocking=True)
            self.lm_head = hf_glm.transformer.output_layer.weight.detach().to(first_device, non_blocking=True)

            self.norm_weight = hf_glm.transformer.encoder.final_layernorm.weight.detach().to(first_device, non_blocking=True)
            self.norm_variance_epsilon = hf_glm.transformer.encoder.final_layernorm.eps

            self.position_ids = torch.arange(0, self.max_length).to(first_device, non_blocking=True)
            self.rotary_cache = self._set_rotary_cache(hf_glm.transformer.rotary_pos_emb, first_device)

            self.layers = []
            for ldx, hf_glm_layer in enumerate(hf_glm.transformer.encoder.layers):
                glm_layer = GlmLayer(ldx, device=self.layer_mapping[str(ldx)])
                glm_layer.init_layer(hf_glm_layer)
                self.layers.append(glm_layer)
                hf_glm.transformer.encoder.layers[ldx] = None

        del hf_glm
        gc.collect()
        torch.cuda.empty_cache()

    def init_kv_cache(self, real_input_length, valid_start, attn_config=None):
        if attn_config is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_dir = os.path.join(project_root, "config")
            model_name = self.model_name.split("/")[-1] + '.json'
            config_file = os.path.join(config_dir, model_name)

            with open(config_file, "r") as f:
                glm_config = json.load(f)
        else:
            glm_config = attn_config

        if self.attention_type == 'Full_Flash_Attn':
            self.kv_cache = flash_attn_cache(
                valid_start=valid_start,
                layer_num=self.num_layers,
                batch_size=self.batch_size,
                max_length=self.max_new_length + real_input_length,
                num_key_value_heads=self.num_key_value_heads,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                layer_mapping=self.layer_mapping,
                num_gpus=self.num_gpus,
                model_size=int(re.search(r'(\d+)[Bb]', self.model_name).group(1))
            )
        elif self.attention_type == 'RetroInfer':
            retroinfer_config = glm_config.get(self.attention_type)
            self.kv_cache = retroinfer_cache(
                valid_start=valid_start,
                layer_num=self.num_layers,
                batch_size=self.batch_size,
                max_length=self.max_new_length + real_input_length,
                num_key_value_heads=self.num_key_value_heads,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                layer_mapping=self.layer_mapping,
                max_new_length=self.max_new_length,
                static_pattern_start=retroinfer_config["static_pattern_start"],
                static_pattern_end=retroinfer_config["static_pattern_end"],
                core=retroinfer_config["core"],
                n_centroids=retroinfer_config["n_centroids"],
                n_segment=retroinfer_config["n_segment"],
                nprobe=retroinfer_config["nprobe"],
                max_compute_cluster_num=retroinfer_config["max_compute_cluster_num"],
                cache_unit_size=retroinfer_config["cache_unit_size"],
                cache_cluster_num=retroinfer_config["cache_cluster_num"],
                num_gpus=self.num_gpus,
                model_size=int(re.search(r'(\d+)[Bb]', self.model_name).group(1)),
                RECALL=self.RECALL
            )
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

    def move(self):
        torch.cuda.empty_cache()
        if self.attention_type == 'Full_Flash_Attn':
            self.kv_cache.move_gpu()
        elif self.attention_type == 'RetroInfer':
            self.kv_cache.prepare_cache()
        torch.cuda.empty_cache()

    def word_embedding(self, inputs_id):
        hidden_states = F.embedding(inputs_id, self.embed_tokens)
        return hidden_states

    def lm(self, hidden_states):
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv, layer.bqkv)
        if self.num_key_value_heads != self.num_heads:
            split_sizes = [
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ]
        else:
            split_sizes = [
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim,
            ]
        query_states, key_states, value_states = qkv.split(split_sizes, dim=-1)
        return query_states, key_states, value_states

    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        hidden_states = F.linear(hidden_states, layer.wo, layer.bo)
        return hidden_states

    def prefill_attention(self, query_states, key_states, value_states, layer_idx):
        if self.prefill_method == 'Full_Flash_Attn':
            attn_out = prefill_full_flash_attn(query_states, key_states, value_states, causal=True)
        elif self.prefill_method == "minfer":
            attn_out = prefill_minfer(query_states, key_states, value_states, self.best_patterns[layer_idx])
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        if self.attention_type == 'Full_Flash_Attn':
            attn_out = decode_full_flash_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        elif self.attention_type == 'RetroInfer':
            attn_out = retroinfer_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out

    def mlp(self, hidden_states, layer):
        hidden_states = F.linear(hidden_states, layer.gate_up_proj, layer.gate_up_proj_bias)
        dim = hidden_states.shape[-1] // 2
        hidden_shape = hidden_states.shape[:-1] + (dim,)
        out = torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        flashinfer.activation.silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, layer.down_proj, layer.down_proj_bias)
        return hidden_states

    def parameter_move(self, hidden_states, ldx):
        next_device = self.layer_mapping[str(ldx+1)] if str(ldx+1) in self.layer_mapping else self.layer_mapping[str(0)]
        torch.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.rotary_cache = self.rotary_cache.to(next_device)
        if self.attention_type == 'Full_Flash_Attn':
            if hidden_states.shape[1] == 1:
                self.kv_cache.batch_indices = self.kv_cache.batch_indices.to(next_device)
                self.kv_cache.valid_length = self.kv_cache.valid_length.to(next_device)
        elif self.attention_type == 'RetroInfer':
            if hidden_states.shape[1] == 1:
                self.kv_cache.gemm_o = self.kv_cache.gemm_o.to(next_device)
                self.kv_cache.softmax_o = self.kv_cache.softmax_o.to(next_device)
                self.kv_cache.norm = self.kv_cache.norm.to(next_device)
                self.kv_cache.sum = self.kv_cache.sum.to(next_device)
                self.kv_cache.es_centroids = self.kv_cache.es_centroids.to(next_device)
                self.kv_cache.es_value_sum = self.kv_cache.es_value_sum.to(next_device)
                self.kv_cache.es_cluster_size = self.kv_cache.es_cluster_size.to(next_device)
                self.kv_cache.execution_buffer_keys = self.kv_cache.execution_buffer_keys.to(next_device)
                self.kv_cache.execution_buffer_values = self.kv_cache.execution_buffer_values.to(next_device)
                self.kv_cache.valid_lengths = self.kv_cache.valid_lengths.to(next_device)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return hidden_states

    def layernorm(self, hidden_states, epsilon, weight):
        if self.use_rmsnorm:
            bsz, seq_len, dim = hidden_states.shape
            hidden_states = hidden_states.reshape(bsz * seq_len, dim)
            hidden_states = flashinfer.rmsnorm(hidden_states, weight, epsilon)
            hidden_states = hidden_states.reshape(bsz, seq_len, dim)
            return hidden_states
        return F.layer_norm(hidden_states, (hidden_states.shape[-1],), weight=weight, bias=None, eps=epsilon)

    def apply_rotary_pos_emb(self, hidden_states, rope_cache):
        bsz, num_heads, seq_len, head_dim = hidden_states.shape
        rot_dim = rope_cache.shape[-2] * 2
        hidden_states, hidden_states_pass = hidden_states[..., :rot_dim], hidden_states[..., rot_dim:]
        rope_cache = rope_cache[:, :seq_len]
        hidden_states = hidden_states.reshape(bsz, num_heads, seq_len, rot_dim // 2, 2)
        rope_cache = rope_cache.view(bsz, 1, seq_len, hidden_states.size(3), 2)
        rotated = torch.stack(
            [
                hidden_states[..., 0] * rope_cache[..., 0] - hidden_states[..., 1] * rope_cache[..., 1],
                hidden_states[..., 1] * rope_cache[..., 0] + hidden_states[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        ).flatten(3)
        return torch.cat((rotated, hidden_states_pass), dim=-1)

    def position_embedd(self, query_states, key_states):
        bsz, seq_len, _ = key_states.shape
        position_ids = self.position_ids[self.kv_cache.context:self.kv_cache.context+seq_len].unsqueeze(0).repeat(bsz, 1)
        rope_cache = self.rotary_cache[position_ids]

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()

        query_states = self.apply_rotary_pos_emb(query_states, rope_cache)
        key_states = self.apply_rotary_pos_emb(key_states, rope_cache)

        query_states = query_states.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        key_states = key_states.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_key_value_heads * self.head_dim)
        return query_states, key_states
