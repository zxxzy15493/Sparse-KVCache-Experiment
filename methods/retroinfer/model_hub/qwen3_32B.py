"""RetroInfer adapter for dense Qwen3 causal language models.

Qwen3 is not interchangeable with the Qwen2/Qwen2.5 adapter: its attention
projections are bias-free and it applies RMSNorm independently to every query
and key head before RoPE.  This module keeps the common RetroInfer execution
loop in :mod:`model_hub.LLM`, while implementing all Qwen3-specific model
operations locally.
"""

import gc
import json
import math
import os

import flashinfer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from attn_hub import (
    decode_full_flash_attn,
    prefill_full_flash_attn,
    retroinfer_decode_attn,
)
from cache_hub import flash_attn_cache, retroinfer_cache

from .LLM import LLM


def _qwen3_transformers_classes():
    """Load Qwen3 classes lazily without breaking existing Qwen2 users.

    The repository's original dependency pin predates Qwen3.  Delaying this
    import means ``from model_hub import QwenModel`` remains valid there, while
    a user who instantiates this adapter receives an actionable error.
    """
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on the runtime install
        raise ImportError(
            "Qwen3 requires a Transformers release with Qwen3 support "
            "(transformers>=4.51.0)."
        ) from exc
    return Qwen3Config, Qwen3ForCausalLM


class Qwen3Layer:
    """The weights needed by one dense Qwen3 decoder layer."""

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device

    def init_layer(self, hf_qwen_layer):
        attention = hf_qwen_layer.self_attn

        q_weight = attention.q_proj.weight.detach()
        k_weight = attention.k_proj.weight.detach()
        v_weight = attention.v_proj.weight.detach()
        self.wqkv = torch.cat((q_weight, k_weight, v_weight), dim=0).to(
            self.device, non_blocking=True
        )

        # Official dense Qwen3 checkpoints have attention_bias=false.  Keep
        # the optional-bias path so
        # a checkpoint with all three projection biases remains well-defined.
        qkv_biases = (attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias)
        if all(bias is None for bias in qkv_biases):
            self.bqkv = None
        elif all(bias is not None for bias in qkv_biases):
            self.bqkv = torch.cat(
                tuple(bias.detach() for bias in qkv_biases), dim=0
            ).to(self.device, non_blocking=True)
        else:
            raise ValueError("Qwen3 attention must use either all or no Q/K/V biases.")

        self.wo = attention.o_proj.weight.detach().to(self.device, non_blocking=True)

        gate_weight = hf_qwen_layer.mlp.gate_proj.weight.detach()
        up_weight = hf_qwen_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = torch.cat((gate_weight, up_weight), dim=0).to(
            self.device, non_blocking=True
        )
        self.down_proj = hf_qwen_layer.mlp.down_proj.weight.detach().to(
            self.device, non_blocking=True
        )

        self.input_layernorm_weight = hf_qwen_layer.input_layernorm.weight.detach().to(
            self.device, non_blocking=True
        )
        self.input_layernorm_variance_epsilon = (
            hf_qwen_layer.input_layernorm.variance_epsilon
        )
        self.post_attention_layernorm_weight = (
            hf_qwen_layer.post_attention_layernorm.weight.detach().to(
                self.device, non_blocking=True
            )
        )
        self.post_attention_layernorm_variance_epsilon = (
            hf_qwen_layer.post_attention_layernorm.variance_epsilon
        )

        # Unlike Qwen2/Qwen2.5, Qwen3 normalizes Q and K per attention head
        # after projection and before rotary position embedding.
        self.q_norm_weight = attention.q_norm.weight.detach().to(
            self.device, non_blocking=True
        )
        self.q_norm_variance_epsilon = attention.q_norm.variance_epsilon
        self.k_norm_weight = attention.k_norm.weight.detach().to(
            self.device, non_blocking=True
        )
        self.k_norm_variance_epsilon = attention.k_norm.variance_epsilon


class Qwen3Model(LLM):
    """Dense Qwen3 implementation for the RetroInfer ``LLM`` interface.

    Architectural dimensions are read from the checkpoint configuration, so
    this adapter supports the dense Qwen3 sizes (for example 0.6B through
    32B).  Qwen3 MoE/VL checkpoints deliberately remain unsupported because
    they use different model classes and decoder-layer structures.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        RECALL: bool = False,
        budget: int = 128,
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map, RECALL, budget)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        qwen3_config_class, _ = _qwen3_transformers_classes()
        self.config = qwen3_config_class.from_pretrained(model_name)
        if self.config.model_type != "qwen3":
            raise ValueError(
                "Qwen3Model supports dense Qwen3 checkpoints only; "
                f"got model_type={self.config.model_type!r}."
            )

        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = getattr(
            self.config, "head_dim", self.hidden_size // self.num_heads
        )
        self.max_position_embeddings = self.config.max_position_embeddings
        rope_scaling = self.config.rope_scaling or {}
        self.original_max_position_embeddings = rope_scaling.get(
            "original_max_position_embeddings", self.max_position_embeddings
        )
        self.yarn_factor = rope_scaling.get("factor", 1)
        # 保留入口或 checkpoint 明确指定的 RoPE；仅在未指定时自动扩展。
        if not rope_scaling and self.max_length > self.original_max_position_embeddings:
            # 未显式指定 RoPE 时，按 checkpoint 的原始窗口扩展到请求上下文。
            self.yarn_factor = math.ceil(
                self.max_length / self.original_max_position_embeddings
            )
            self.config.rope_scaling = {
                "rope_type": "yarn",
                "factor": float(self.yarn_factor),
                "original_max_position_embeddings": self.original_max_position_embeddings,
            }
        self.vocab_size = self.config.vocab_size

        eos_token_id = getattr(self.config, "eos_token_id", None)
        eos_tokens = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        eos_tokens.extend(
            (getattr(self.tokenizer, "eos_token_id", None), getattr(self.tokenizer, "im_end_id", None))
        )
        self.eos_tokens = list(dict.fromkeys(token for token in eos_tokens if token is not None))

        self.init_model()

    def _set_cos_sin_cache(self):
        """Build the RoPE cache from the Qwen3 rotary embedding parameters."""
        positions = torch.arange(
            self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        frequencies = torch.outer(positions, self.inv_freq)
        return (
            frequencies.cos() * self.attention_scaling,
            frequencies.sin() * self.attention_scaling,
        )

    def init_model(self):
        _, qwen3_for_causal_lm = _qwen3_transformers_classes()
        hf_qwen = qwen3_for_causal_lm.from_pretrained(
            self.model_name, config=self.config, dtype=self.dtype
        )

        # Cache allocation uses this as a model-weight memory estimate.  Do
        # not retain the former 32B constant: it underestimates free memory
        # for larger models and vastly overestimates it for smaller Qwen3s.
        self.model_size_b = sum(parameter.numel() for parameter in hf_qwen.parameters()) / 1e9

        self.num_gpus = torch.cuda.device_count() if self.device_map == "auto" else 1
        if self.device_map == "auto" and self.num_gpus == 1:
            self.device_map = "cuda:0"

        if self.device_map != "auto":
            self.layer_mapping = {
                str(layer_idx): self.device_map for layer_idx in range(self.num_layers)
            }
            root_device = self.device_map
        else:
            self.gpu_ids = list(range(self.num_gpus))
            self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
            self.layer_mapping = {
                str(layer_idx): f"cuda:{layer_idx // self.layer_interval}"
                for layer_idx in range(self.num_layers)
            }
            root_device = f"cuda:{self.gpu_ids[0]}"

        self.embed_tokens = hf_qwen.model.embed_tokens.weight.detach().to(
            root_device, non_blocking=True
        )
        self.lm_head = hf_qwen.lm_head.weight.detach().to(root_device, non_blocking=True)
        self.norm_weight = hf_qwen.model.norm.weight.detach().to(
            root_device, non_blocking=True
        )
        self.norm_variance_epsilon = hf_qwen.model.norm.variance_epsilon

        self.position_ids = torch.arange(0, self.max_length).to(
            root_device, non_blocking=True
        )
        self.inv_freq = hf_qwen.model.rotary_emb.inv_freq.detach().to(
            root_device, non_blocking=True
        )
        self.attention_scaling = hf_qwen.model.rotary_emb.attention_scaling
        self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
        self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

        self.layers = []
        for layer_idx, hf_qwen_layer in enumerate(hf_qwen.model.layers):
            layer = Qwen3Layer(
                layer_idx, device=self.layer_mapping[str(layer_idx)]
            )
            layer.init_layer(hf_qwen_layer)
            self.layers.append(layer)
            hf_qwen.model.layers[layer_idx] = None

        del self.inv_freq, self.cos_cache, self.sin_cache, hf_qwen
        gc.collect()
        torch.cuda.empty_cache()

    def _per_head_rmsnorm(self, states, num_heads, weight, epsilon):
        """Apply Qwen3 Q/K RMSNorm to the final dimension of every head."""
        batch_size, seq_len, _ = states.shape
        states = states.reshape(batch_size * seq_len * num_heads, self.head_dim).contiguous()
        states = flashinfer.rmsnorm(states, weight, epsilon)
        return states.reshape(batch_size, seq_len, num_heads * self.head_dim)

    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv, layer.bqkv)
        query_states, key_states, value_states = qkv.split(
            (
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ),
            dim=-1,
        )
        query_states = self._per_head_rmsnorm(
            query_states,
            self.num_heads,
            layer.q_norm_weight,
            layer.q_norm_variance_epsilon,
        )
        key_states = self._per_head_rmsnorm(
            key_states,
            self.num_key_value_heads,
            layer.k_norm_weight,
            layer.k_norm_variance_epsilon,
        )
        return query_states, key_states, value_states

    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        """Project Qwen3's attention-width output back to hidden size.

        Qwen2 has ``num_heads * head_dim == hidden_size``, which made the
        inherited implementation's reshape appear universal.  Some Qwen3
        sizes have a wider attention representation than residual stream, so
        the output projection must reshape using the attention width first.
        """
        attention_dim = self.num_heads * self.head_dim
        hidden_states = hidden_states.reshape(bsz, seq_len, attention_dim)
        return F.linear(hidden_states, layer.wo)

    def prefill_attention(self, query_states, key_states, value_states, layer_idx):
        if self.prefill_method == "Full_Flash_Attn":
            return prefill_full_flash_attn(
                query_states, key_states, value_states, causal=True
            )
        if self.prefill_method == "minfer":
            raise NotImplementedError(
                "Qwen3 has no calibrated MInference pattern profile. "
                "Use prefill_method='Full_Flash_Attn'."
            )
        raise ValueError(f"Unsupported prefill method: {self.prefill_method}")

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        if self.attention_type == "Full_Flash_Attn":
            return decode_full_flash_attn(
                query_states, key_states, value_states, layer_idx, self.kv_cache
            )
        if self.attention_type == "RetroInfer":
            return retroinfer_decode_attn(
                query_states, key_states, value_states, layer_idx, self.kv_cache
            )
        raise ValueError(f"Unsupported attention type: {self.attention_type}")

    def move(self):
        torch.cuda.empty_cache()
        if self.attention_type == "Full_Flash_Attn":
            self.kv_cache.move_gpu()
        elif self.attention_type == "RetroInfer":
            self.kv_cache.prepare_cache()
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        torch.cuda.empty_cache()

    def word_embedding(self, input_ids):
        return F.embedding(input_ids, self.embed_tokens)

    def lm(self, hidden_states):
        return F.linear(hidden_states, self.lm_head).float()

    def mlp(self, hidden_states, layer):
        gate_and_up = F.linear(hidden_states, layer.gate_up_proj)
        intermediate_size = gate_and_up.shape[-1] // 2
        activated = torch.empty(
            gate_and_up.shape[:-1] + (intermediate_size,),
            dtype=gate_and_up.dtype,
            device=gate_and_up.device,
        )
        flashinfer.activation.silu_and_mul(gate_and_up, activated)
        return F.linear(activated, layer.down_proj)

    def parameter_move(self, hidden_states, layer_idx):
        next_layer = str(layer_idx + 1)
        next_device = self.layer_mapping.get(next_layer, self.layer_mapping["0"])
        torch.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)

        if hidden_states.shape[1] != 1:
            return hidden_states
        if self.attention_type == "Full_Flash_Attn":
            self.kv_cache.batch_indices = self.kv_cache.batch_indices.to(next_device)
            self.kv_cache.valid_length = self.kv_cache.valid_length.to(next_device)
        elif self.attention_type == "RetroInfer":
            for name in (
                "gemm_o",
                "softmax_o",
                "norm",
                "sum",
                "es_centroids",
                "es_value_sum",
                "es_cluster_size",
                "execution_buffer_keys",
                "execution_buffer_values",
                "valid_lengths",
            ):
                setattr(self.kv_cache, name, getattr(self.kv_cache, name).to(next_device))
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return hidden_states

    def layernorm(self, hidden_states, epsilon, weight):
        batch_size, seq_len, hidden_size = hidden_states.shape
        normalized = flashinfer.rmsnorm(
            hidden_states.reshape(batch_size * seq_len, hidden_size), weight, epsilon
        )
        return normalized.reshape(batch_size, seq_len, hidden_size)

    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        batch_size, _, query_dim = query_states.shape
        _, _, key_dim = key_states.shape
        query_states = query_states.view(-1, query_dim)
        key_states = key_states.view(-1, key_dim)
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            position_ids,
            query_states,
            key_states,
            self.head_dim,
            self.cos_sin_cache,
            True,
        )
        return (
            query_states.view(batch_size, -1, query_dim),
            key_states.view(batch_size, -1, key_dim),
        )

    def position_embedd(self, query_states, key_states):
        batch_size, seq_len, _ = key_states.shape
        position_ids = self.position_ids[
            self.kv_cache.context : self.kv_cache.context + seq_len
        ].unsqueeze(0).repeat(batch_size, 1)
        return self.apply_rotary_pos_emb(query_states, key_states, position_ids)

    def init_kv_cache(self, real_input_length, valid_start, attn_config=None):
        if attn_config is None and self.attention_type == "RetroInfer":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(
                os.path.dirname(current_dir), "config", "Qwen3.json"
            )
            with open(config_file, "r", encoding="utf-8") as file:
                qwen_config = json.load(file)
        elif attn_config is not None:
            qwen_config = attn_config

        if self.attention_type == "Full_Flash_Attn":
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
                model_size=self.model_size_b,
            )
        elif self.attention_type == "RetroInfer":
            retroinfer_config = qwen_config["RetroInfer"]
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
                model_size=self.model_size_b,
                RECALL=self.RECALL,
                budget=self.budget,
                buffer_multiplier=retroinfer_config.get("buffer_multiplier", 4),
            )
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")


# Backward-compatible names for existing callers and benchmark scripts.
Qwen3_32BLayer = Qwen3Layer
Qwen3_32BModel = Qwen3Model
