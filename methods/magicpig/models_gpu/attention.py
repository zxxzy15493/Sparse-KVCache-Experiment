"""MagicPIG attention adapted from MagicPIG/models/hf/qwen.py.

Uses native Transformers Qwen2/Qwen3 models and caches; no FlashInfer,
LSH extension, or AVX-512 CPU kernel is imported. Padding is excluded from
historical-key statistics and each request retains its own sink/window.
"""

import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask


class MagicPIGState:
    chunk_size = 2048

    def __init__(self, K, L, sink_tokens, window):
        self.K = K
        self.L = L
        self.sink_tokens = int(sink_tokens)
        self.window = int(window)
        if not 0 <= K <= 31 or L < 1 or self.sink_tokens < 0 or self.window < 0:
            raise ValueError("Qwen MagicPIG requires 0 <= K <= 31, L >= 1, S >= 0 and W >= 0")
        self.hash_dtype = torch.uint8 if K <= 8 else torch.int16 if K <= 15 else torch.int32
        self.reset()

    def reset(self):
        self.clear_cache()
        self.sparse = 0.0
        self.sampled_tokens = 0.0
        self.decode_tokens = 0

    def clear_cache(self):
        """Release per-request tensors while preserving reported counters."""
        self.layers = {}
        self.hash_matrices = {}
        self.prompt_mask = None
        self.is_prefill = False

    @property
    def workload(self):
        return self.sparse

    def clear(self):
        self.clear_cache()

    def hash_codes(self, keys, matrix):
        codes = torch.empty(*keys.shape[:-1], self.L, dtype=self.hash_dtype, device=keys.device)
        for start in range(0, keys.shape[-2], self.chunk_size):
            stop = min(start + self.chunk_size, keys.shape[-2])
            chunk = keys[..., start:stop, :]
            bits = (chunk @ matrix).reshape(*chunk.shape[:-1], self.L, self.K).gt(0)
            packed = torch.zeros(bits.shape[:-1], dtype=torch.int32, device=keys.device)
            for bit in range(self.K):
                packed.bitwise_or_(bits[..., bit].int() << bit)
            codes[..., start:stop, :].copy_(packed)
        return codes

    def collision_mask(self, query_codes, key_codes):
        # Keep the comparison and its integer reduction bounded by chunk_size.
        mask = torch.empty(
            *query_codes.shape[:-1], key_codes.shape[-2],
            dtype=torch.bool, device=query_codes.device,
        )
        for start in range(0, key_codes.shape[-2], self.chunk_size):
            stop = min(start + self.chunk_size, key_codes.shape[-2])
            matches = query_codes.unsqueeze(-2) == key_codes[..., None, start:stop, :]
            mask[..., start:stop] = matches.sum(dim=-1, dtype=torch.int32).ge(2)
        return mask

    def prepare_layer(self, module, key, value):
        prefix_len = self.prompt_mask.shape[-1] if self.prompt_mask is not None else key.shape[-2] - 1
        valid = self.prompt_mask
        if valid is None:
            valid = torch.ones(key.shape[0], prefix_len, dtype=torch.bool, device=key.device)
        else:
            valid = valid.to(device=key.device, dtype=torch.bool)
        if valid.shape != (key.shape[0], prefix_len):
            raise ValueError("MagicPIG prompt mask does not match the native KV cache")
        positions = valid.long().cumsum(dim=-1) - 1
        lengths = valid.sum(dim=-1, keepdim=True)
        history = valid & (positions >= self.sink_tokens) & (positions < lengths - self.window)
        counts = history.sum(dim=-1)
        if not history.any():
            return None
        matrix_id = (module.layer_idx // 16, key.device, key.dtype)
        if matrix_id not in self.hash_matrices:
            self.hash_matrices[matrix_id] = torch.randn(
                1, key.shape[1], key.shape[-1], self.K * self.L,
                dtype=key.dtype, device=key.device,
            )
        matrix = self.hash_matrices[matrix_id]
        prefix = key[..., :prefix_len, :]
        average = (prefix.float() * history[:, None, :, None]).sum(dim=-2, keepdim=True)
        average = (average / counts[:, None, None, None].clamp_min(1)).to(key.dtype)
        key_norm = torch.empty(*key.shape[:2], prefix_len, dtype=torch.float32, device=key.device)
        codes = torch.empty(*key.shape[:2], prefix_len, self.L, dtype=self.hash_dtype, device=key.device)
        for start in range(0, prefix_len, self.chunk_size):
            stop = min(start + self.chunk_size, prefix_len)
            centered = prefix[..., start:stop, :] - average
            key_norm[..., start:stop] = centered.norm(p=2, dim=-1).float()
            codes[..., start:stop, :] = self.hash_codes(centered, matrix)
        state = {
            "end": prefix_len, "average": average, "key_norm": key_norm,
            "hash_codes": codes, "matrix": matrix,
            "valid": valid, "history": history, "counts": counts,
        }
        self.layers[module.layer_idx] = state
        return state


def _apply_attention_mask(scores, mask):
    if mask is None:
        return scores
    if mask.dtype == torch.bool:
        return scores.masked_fill(~mask, -torch.inf)
    return scores + mask


def magicpig_attention_forward(module, query, key, value, attention_mask,
                              dropout=0.0, scaling=None, **kwargs):
    manager = module.magicpig_state
    if manager.is_prefill or query.shape[-2] != 1 or module.layer_idx % 16 == 0 or manager.K == 0:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )
    state = manager.layers.get(module.layer_idx)
    if state is None:
        state = manager.prepare_layer(module, key, value)
    if state is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )
    batch, query_heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    groups = query_heads // kv_heads
    grouped_query = query.reshape(batch, kv_heads, groups, head_dim)
    end = state["end"]
    history = state["history"][:, None, None, :]
    normalized = query / query.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
    query_codes = manager.hash_codes(
        normalized.reshape(batch, kv_heads, groups, head_dim), state["matrix"],
    )
    collisions = manager.collision_mask(query_codes, state["hash_codes"]) & history
    eligible = state["counts"] > 0
    sampled = collisions.float().sum(dim=-1)
    manager.sampled_tokens += sampled[eligible].mean().item()
    manager.sparse += (sampled[eligible] / state["counts"][eligible, None, None]).mean().item()
    manager.decode_tokens += 1

    # A shared key shift cancels in softmax. GQA query groups share native KV.
    centered = key - state["average"]
    scores = (grouped_query @ centered.transpose(-2, -1)).float()
    scale = scaling if scaling is not None else head_dim ** -0.5
    cosine = (scores[..., :end] / (
        grouped_query.norm(p=2, dim=-1, keepdim=True).float()
        * state["key_norm"].unsqueeze(-2)
    ).clamp_min(1e-6)).clamp(-1, 1)
    probability = (1 - cosine.acos() / torch.pi) ** manager.K
    inclusion = (1 - (1 - probability) ** manager.L
                 - manager.L * (1 - probability) ** (manager.L - 1) * probability).clamp_min(0)
    scores = scores * scale
    scores[..., :end] -= torch.where(history, (inclusion + 1e-4).log(), 0)
    scores[..., :end].masked_fill_(history & ~collisions, -torch.inf)
    scores[..., :end].masked_fill_(~state["valid"][:, None, None, :], -torch.inf)
    scores = _apply_attention_mask(scores.reshape(batch, query_heads, 1, -1), attention_mask)
    weights = scores.softmax(dim=-1).nan_to_num(nan=0.0).to(query.dtype)
    output = (weights.reshape(batch, kv_heads, groups, -1) @ value)
    return output.reshape(batch, query_heads, 1, head_dim).transpose(1, 2).contiguous(), None


def register_qwen_magicpig():
    name = "magicpig_gpu_qwen"
    AttentionInterface.register(name, magicpig_attention_forward)
    AttentionMaskInterface.register(name, sdpa_mask)
    return name


def configure_qwen_magicpig(model, K, L, sink_tokens, window):
    manager = MagicPIGState(K, L, sink_tokens, window)
    for layer in model.model.layers:
        layer.self_attn.magicpig_state = manager
    return manager
