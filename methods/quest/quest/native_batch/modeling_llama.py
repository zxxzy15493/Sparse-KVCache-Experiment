"""使用 Quest 原生算子实现等长批处理。"""

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

from .controller import NativeBatchController
from .ops import rms_norm


class NativeRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class NativeAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = getattr(config, 'head_dim', config.hidden_size // self.q_heads)
        bias = getattr(config, 'attention_bias', False)
        self.q_proj = nn.Linear(config.hidden_size, self.q_heads * self.dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.dim, bias=bias)
        self.o_proj = nn.Linear(self.q_heads * self.dim, config.hidden_size, bias=bias)
        if config.model_type == 'qwen3':
            self.q_norm = NativeRMSNorm(self.dim, config.rms_norm_eps)
            self.k_norm = NativeRMSNorm(self.dim, config.rms_norm_eps)
        self.rope_theta = config.rope_theta
        self.rope_scale = (config.rope_scaling or {}).get('factor', 1.0)

    def forward(self, x, controller, skip_layers):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.q_heads, self.dim)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.dim)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.dim)
        if hasattr(self, 'q_norm'):
            q = self.q_norm(q.flatten(1, 2)).view(batch, length, self.q_heads, self.dim)
            k = self.k_norm(k.flatten(1, 2)).view(batch, length, self.kv_heads, self.dim)
        if controller.q_heads != self.q_heads:
            # Keep the original native GQA group sizes while preserving head order.
            group = self.q_heads // self.kv_heads
            padded_group = controller.q_heads // self.kv_heads
            padded = q.new_zeros(batch, length, self.kv_heads, padded_group, self.dim)
            padded[:, :, :, :group] = q.view(batch, length, self.kv_heads, group, self.dim)
            q = padded.flatten(2, 3)
        q, k = q.contiguous(), k.contiguous()
        controller.kernels.rope(q, k, controller.position, self.rope_scale, self.rope_theta)
        controller.append(self.layer_idx, k, v)
        output = controller.attention(self.layer_idx, q, dense=self.layer_idx < skip_layers)
        if controller.q_heads != self.q_heads:
            group = self.q_heads // self.kv_heads
            padded_group = controller.q_heads // self.kv_heads
            output = output.view(batch, length, self.kv_heads, padded_group, self.dim)[:, :, :, :group]
        return self.o_proj(output.reshape(batch, length, self.q_heads * self.dim))


class NativeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        bias = getattr(config, 'mlp_bias', False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NativeDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = NativeAttention(config, index)
        self.mlp = NativeMLP(config)
        self.input_layernorm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, controller, skip_layers):
        hidden = x + self.self_attn(self.input_layernorm(x), controller, skip_layers)
        # Match the original Quest path's bounded post-attention FFN temporary.
        for start in range(0, hidden.size(1), 1024):
            chunk = hidden[:, start:start + 1024]
            chunk.add_(self.mlp(self.post_attention_layernorm(chunk)))
        return hidden


class NativeLlamaBody(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([NativeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)


class _NativeBatchCausalLM:
    _tied_weights_keys = ['lm_head.weight']
    _supports_sdpa = False

    def _initialize_native_model(self, config):
        if getattr(config, 'pretraining_tp', 1) != 1:
            raise ValueError('Native batched Quest requires pretraining_tp=1')
        self.model = NativeLlamaBody(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.controller = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def quest_init(self, batch_size, max_seq_len, token_budget, page_size=16,
                   skip_layers=2, legacy_head_selection=True):
        device = self.model.embed_tokens.weight.device
        if device.type != 'cuda' or any(p.device != device or p.dtype != torch.float16
                                       for p in self.parameters()):
            raise ValueError('Load all model weights as float16 on one CUDA device first')
        if self.controller is not None:
            self.controller.clear()
            self.controller = None
        self.skip_layers = skip_layers
        attention = self.model.layers[0].self_attn
        self.controller = NativeBatchController(len(self.model.layers), batch_size,
            attention.q_heads, attention.kv_heads, attention.dim, max_seq_len,
            token_budget, page_size, device, legacy_head_selection)
        return self

    def quest_clear(self):
        if self.controller is not None:
            self.controller.clear()

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=True, logits_to_keep=1, return_dict=True, **kwargs):
        controller = self.controller
        if controller is None:
            raise RuntimeError('Call quest_init before inference')
        if input_ids.ndim != 2 or input_ids.size(0) != controller.batch_size or input_ids.size(1) == 0:
            raise ValueError('Expected nonempty input_ids with the initialized batch size')
        if attention_mask is not None and (attention_mask.ndim != 2 or
                attention_mask.size(0) != input_ids.size(0) or not bool((attention_mask == 1).all())):
            raise ValueError('Only equal-length unpadded requests are supported')
        if past_key_values is not None or not use_cache:
            raise ValueError('Native Quest owns the cache; use_cache=True without external past_key_values')
        if kwargs.get('labels') is not None:
            raise ValueError('This model supports inference only')
        keep = kwargs.get('num_logits_to_keep', logits_to_keep)
        try:
            hidden = self.model.embed_tokens(input_ids)
            controller.prepare(input_ids.size(1), dense=self.skip_layers > 0)
            for index, layer in enumerate(self.model.layers):
                if self.skip_layers > 0 and index == self.skip_layers:
                    controller.end_forward()
                    controller.begin_forward(dense=False, update_tensors=False)
                hidden = layer(hidden, controller, self.skip_layers)
            controller.end_forward()
            hidden = self.model.norm(hidden)
            hidden = hidden[:, -keep:] if isinstance(keep, int) and keep else hidden
            logits = self.lm_head(hidden).float()
        except Exception:
            self.quest_clear()
            raise
        if not return_dict:
            return (logits,)
        return CausalLMOutputWithPast(logits=logits)

    @torch.inference_mode()
    def generate_fixed(self, input_ids, max_new_tokens, prefill_hook=None, decode_hook=None):
        """Return exactly max_new_tokens greedy tokens, ignoring EOS."""
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        self.quest_clear()
        output = self(input_ids)
        if prefill_hook is not None:
            prefill_hook()
        tokens = [output.logits[:, -1].argmax(-1, keepdim=True)]
        for _ in range(max_new_tokens - 1):
            output = self(tokens[-1])
            tokens.append(output.logits[:, -1].argmax(-1, keepdim=True))
        if decode_hook is not None:
            decode_hook()
        return torch.cat(tokens, dim=1)


class NativeBatchLlamaForCausalLM(_NativeBatchCausalLM, LlamaPreTrainedModel):
    """Llama 检查点的 Quest 原生推理模型。"""

    def __init__(self, config):
        super().__init__(config)
        self._initialize_native_model(config)


class NativeBatchQwen2ForCausalLM(_NativeBatchCausalLM, Qwen2PreTrainedModel):
    """Qwen2 和 Qwen2.5 检查点的 Quest 原生推理模型。"""

    def __init__(self, config):
        super().__init__(config)
        self._initialize_native_model(config)


class NativeBatchQwen3ForCausalLM(_NativeBatchCausalLM, Qwen3PreTrainedModel):
    """稠密 Qwen3 检查点的 Quest 原生推理模型。"""

    def __init__(self, config):
        super().__init__(config)
        self._initialize_native_model(config)
