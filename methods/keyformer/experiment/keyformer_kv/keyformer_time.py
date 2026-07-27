import math
import types
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

breaktime_registry = []
TOTAL_LAYERS = 0

prefill_start_recorded = False
decode_start_recorded = False

global_prefill_start_event = torch.cuda.Event(enable_timing=True)
global_prefill_end_event = torch.cuda.Event(enable_timing=True)
global_decode_start_event = torch.cuda.Event(enable_timing=True)
global_decode_end_event = torch.cuda.Event(enable_timing=True)

class KeyformerKVCache:
    def __init__(self, total_budget=1024, recent_size=32, tau_init=1.0, tau_delta=0.01, k_seq_dim=2, v_seq_dim=2):
        self.recent_size = recent_size
        self.key_size = total_budget - recent_size
        self.cache_size = total_budget
        self.tau_init = tau_init
        self.tau_delta = tau_delta
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.key_score = None

        self.construct_event_time = 0.0
        self.prefill_writeback_event_time = 0.0
        self.decode_writeback_event_time = 0.0
        self.prefill_attn_event_time = 0.0
        self.decode_attn_event_time = 0.0
        self.retrieve_event_time = 0.0

        self.construct_start_event = torch.cuda.Event(enable_timing=True)
        self.construct_end_event   = torch.cuda.Event(enable_timing=True)
        self.prefill_writeback_start_event = torch.cuda.Event(enable_timing=True)
        self.prefill_writeback_end_event   = torch.cuda.Event(enable_timing=True)
        self.decode_writeback_start_event = torch.cuda.Event(enable_timing=True)
        self.decode_writeback_end_event   = torch.cuda.Event(enable_timing=True)
        self.prefill_attn_start_event = torch.cuda.Event(enable_timing=True)
        self.prefill_attn_end_event   = torch.cuda.Event(enable_timing=True)
        self.decode_attn_start_event = torch.cuda.Event(enable_timing=True)
        self.decode_attn_end_event   = torch.cuda.Event(enable_timing=True)
        self.retrieve_start_event = torch.cuda.Event(enable_timing=True)
        self.retrieve_end_event   = torch.cuda.Event(enable_timing=True)

        self._global_step = 0


def _acc(kv, s, e):
    e.synchronize()
    return s._orig_elapsed(e) / 1000.0


@torch.no_grad()
def common_timing_and_evict(self, query_states, key_states, value_states, past_key_value, bsz, q_len, sin, cos):
    device = query_states.device
    num_heads = query_states.shape[1]
    head_dim = query_states.shape[-1]

    if q_len > 1:
        self.kv_cache.prefill_writeback_start_event.record()
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos})
        self.kv_cache.prefill_writeback_end_event.record()
        self.kv_cache.prefill_writeback_event_time += _acc(self.kv_cache, self.kv_cache.prefill_writeback_start_event, self.kv_cache.prefill_writeback_end_event)

        kv_seq_len = key_states.shape[self.kv_cache.k_seq_dim]

        self.kv_cache.construct_start_event.record()

        current_tau = self.kv_cache.tau_init + self.kv_cache._global_step * self.kv_cache.tau_delta
        current_scores = torch.zeros((bsz, num_heads, kv_seq_len), device=device, dtype=torch.bfloat16)
        chunk_size = 128
        for i in range(0, q_len, chunk_size):
            end_idx = min(i + chunk_size, q_len)
            q_chunk = query_states[:, :, i:end_idx, :]
            aw = torch.matmul(q_chunk, key_states.transpose(-1, -2)) / math.sqrt(head_dim)
            row_indices = torch.arange(i, end_idx, device=device).view(-1, 1)
            col_indices = torch.arange(kv_seq_len, device=device).view(1, -1)
            aw.masked_fill_(~(row_indices >= (col_indices - (kv_seq_len - q_len))).unsqueeze(0).unsqueeze(0), float("-inf"))
            gumbel_w = F.gumbel_softmax(aw, tau=current_tau, hard=False, dim=-1).to(torch.bfloat16)
            current_scores += gumbel_w.sum(dim=-2)

        self.kv_cache.key_score = current_scores

        if kv_seq_len <= self.kv_cache.cache_size:
            k_comp, v_comp = key_states, value_states
        else:
            hb = kv_seq_len - self.kv_cache.recent_size
            keep_recent_idx = torch.arange(hb, kv_seq_len, device=device).view(1, 1, -1).expand(bsz, num_heads, -1)
            history_scores = self.kv_cache.key_score[:, :, :hb]
            _, topk_idx = torch.topk(history_scores, self.kv_cache.key_size, dim=-1)
            topk_idx, _ = torch.sort(topk_idx, dim=-1)
            keep_idx = torch.cat([topk_idx, keep_recent_idx], dim=-1)
            keep_idx, _ = torch.sort(keep_idx, dim=-1)
            gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_comp = torch.gather(key_states, self.kv_cache.k_seq_dim, gather_idx)
            v_comp = torch.gather(value_states, self.kv_cache.v_seq_dim, gather_idx)
            self.kv_cache.key_score = torch.gather(self.kv_cache.key_score, -1, keep_idx)

        self.kv_cache.construct_end_event.record()

        self.kv_cache.prefill_writeback_start_event.record()
        past_key_value.key_cache[self.layer_idx] = k_comp
        past_key_value.value_cache[self.layer_idx] = v_comp
        self.kv_cache.prefill_writeback_end_event.record()
        self.kv_cache.prefill_writeback_event_time += _acc(self.kv_cache, self.kv_cache.prefill_writeback_start_event, self.kv_cache.prefill_writeback_end_event)

        self.kv_cache.prefill_attn_start_event.record()
        attn_out = flash_attn_func(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=True
        ).reshape(bsz, q_len, self.hidden_size)
        self.kv_cache.prefill_attn_end_event.record()

    else:
        self.kv_cache.decode_writeback_start_event.record()
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos})
        self.kv_cache.decode_writeback_end_event.record()
        self.kv_cache.decode_writeback_event_time += _acc(self.kv_cache, self.kv_cache.decode_writeback_start_event, self.kv_cache.decode_writeback_end_event)

        kv_seq_len = key_states.shape[self.kv_cache.k_seq_dim]

        self.kv_cache.retrieve_start_event.record()

        current_tau = self.kv_cache.tau_init + self.kv_cache._global_step * self.kv_cache.tau_delta
        current_scores = torch.zeros((bsz, num_heads, kv_seq_len), device=device, dtype=torch.bfloat16)
        aw = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(head_dim)
        gumbel_w = F.gumbel_softmax(aw, tau=current_tau, hard=False, dim=-1).to(torch.bfloat16)
        current_scores += gumbel_w.sum(dim=-2)
        self.kv_cache.key_score = torch.cat([self.kv_cache.key_score, current_scores[:, :, -1:]], dim=-1)

        if kv_seq_len <= self.kv_cache.cache_size:
            k_comp, v_comp = key_states, value_states
        else:
            hb = kv_seq_len - self.kv_cache.recent_size
            keep_recent_idx = torch.arange(hb, kv_seq_len, device=device).view(1, 1, -1).expand(bsz, num_heads, -1)
            history_scores = self.kv_cache.key_score[:, :, :hb]
            _, topk_idx = torch.topk(history_scores, self.kv_cache.key_size, dim=-1)
            topk_idx, _ = torch.sort(topk_idx, dim=-1)
            keep_idx = torch.cat([topk_idx, keep_recent_idx], dim=-1)
            keep_idx, _ = torch.sort(keep_idx, dim=-1)
            gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_comp = torch.gather(key_states, self.kv_cache.k_seq_dim, gather_idx)
            v_comp = torch.gather(value_states, self.kv_cache.v_seq_dim, gather_idx)
            self.kv_cache.key_score = torch.gather(self.kv_cache.key_score, -1, keep_idx)

        self.kv_cache.retrieve_end_event.record()
        self.kv_cache.retrieve_event_time += _acc(self.kv_cache, self.kv_cache.retrieve_start_event, self.kv_cache.retrieve_end_event)

        self.kv_cache.decode_writeback_start_event.record()
        past_key_value.key_cache[self.layer_idx] = k_comp
        past_key_value.value_cache[self.layer_idx] = v_comp
        self.kv_cache.decode_writeback_end_event.record()
        self.kv_cache.decode_writeback_event_time += _acc(self.kv_cache, self.kv_cache.decode_writeback_start_event, self.kv_cache.decode_writeback_end_event)

        self.kv_cache.decode_attn_start_event.record()
        attn_out = flash_attn_func(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=True
        ).reshape(bsz, q_len, self.hidden_size)
        self.kv_cache.decode_attn_end_event.record()
        self.kv_cache.decode_attn_event_time += _acc(self.kv_cache, self.kv_cache.decode_attn_start_event, self.kv_cache.decode_attn_end_event)

    if self.layer_idx == 0:
        self.kv_cache._global_step += 1

    return self.o_proj(attn_out), None, past_key_value


def mlp_forward_timing_patch(self, x, *args, **kwargs):
    global TOTAL_LAYERS
    q_len = x.shape[1]
    out = self.original_forward(x, *args, **kwargs)
    if q_len > 1:
        if self.layer_idx == (TOTAL_LAYERS - 1):
            global_prefill_end_event.record()
    else:
        if self.layer_idx == (TOTAL_LAYERS - 1):
            global_decode_end_event.record()
    return out


def keyformer_forward_timing_patch(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
    global prefill_start_recorded, decode_start_recorded
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 and not prefill_start_recorded:
        prefill_start_recorded = True
        global_prefill_start_event.record()
    if q_len == 1 and not decode_start_recorded:
        decode_start_recorded = True
        global_decode_start_event.record()

    if "Qwen2" in type(self).__name__:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    return common_timing_and_evict(self, query_states, key_states, value_states, past_key_value, bsz, q_len, sin, cos)


def reset_time_global_state(model):
    global prefill_start_recorded, decode_start_recorded
    prefill_start_recorded = False
    decode_start_recorded = False
    for module in model.modules():
        c = getattr(module, "kv_cache", None)
        if c is not None:
            c.construct_event_time = 0.0
            c.prefill_writeback_event_time = 0.0
            c.decode_writeback_event_time = 0.0
            c.prefill_attn_event_time = 0.0
            c.decode_attn_event_time = 0.0
            c.retrieve_event_time = 0.0
            c.key_score = None
            c._global_step = 0


def enable_keyformer_time(model, key_size=992, recent_size=32, tau_init=1.0, tau_delta=0.01):
    global TOTAL_LAYERS
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except ImportError:
        LlamaDecoderLayer = None
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    except ImportError:
        Qwen2DecoderLayer = None

    total_budget = key_size + recent_size
    num_layers = 0
    for m in model.modules():
        if (LlamaDecoderLayer and isinstance(m, LlamaDecoderLayer)) or (Qwen2DecoderLayer and isinstance(m, Qwen2DecoderLayer)):
            num_layers += 1
    TOTAL_LAYERS = num_layers

    patched = 0
    for name, module in model.named_modules():
        if (LlamaDecoderLayer and isinstance(module, LlamaDecoderLayer)) or (Qwen2DecoderLayer and isinstance(module, Qwen2DecoderLayer)):
            l_idx = int(name.split('.')[-1])
            kv_cache = KeyformerKVCache(total_budget=total_budget, recent_size=recent_size,
                                         tau_init=tau_init, tau_delta=tau_delta, k_seq_dim=2, v_seq_dim=2)

            kv_cache.prefill_writeback_start_event._orig_elapsed = kv_cache.prefill_writeback_start_event.elapsed_time
            kv_cache.decode_writeback_start_event._orig_elapsed = kv_cache.decode_writeback_start_event.elapsed_time
            kv_cache.decode_writeback_start_event.elapsed_time = lambda ee, c=kv_cache: c.decode_writeback_event_time * 1000.0

            kv_cache.decode_attn_start_event._orig_elapsed = kv_cache.decode_attn_start_event.elapsed_time
            kv_cache.decode_attn_start_event.elapsed_time = lambda ee, c=kv_cache: c.decode_attn_event_time * 1000.0

            kv_cache.retrieve_start_event._orig_elapsed = kv_cache.retrieve_start_event.elapsed_time
            kv_cache.retrieve_start_event.elapsed_time = lambda ee, c=kv_cache: c.retrieve_event_time * 1000.0

            attn_module = module.self_attn
            attn_module.kv_cache = kv_cache
            attn_module.layer_idx = l_idx
            attn_module.forward = types.MethodType(keyformer_forward_timing_patch, attn_module)

            mlp_module = module.mlp
            mlp_module.kv_cache = kv_cache
            mlp_module.layer_idx = l_idx
            mlp_module.original_forward = mlp_module.forward
            mlp_module.forward = types.MethodType(mlp_forward_timing_patch, mlp_module)

            patched += 1

    print(f"Keyformer Co-Timing Profiler Enabled. Patched {patched} Layers (Budget={total_budget}, Key={key_size}, Recent={recent_size}).")
