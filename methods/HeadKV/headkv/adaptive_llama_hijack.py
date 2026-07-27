import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Optional, Tuple, Union
import warnings
import math
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    _flash_attention_forward,  # compat 4.45: class method in 4.37, module-level function in 4.45
)
from transformers.utils import (
    logging,
    is_flash_attn_2_available,
)

_HEADKV_FLASH_ATTN_2_AVAILABLE = is_flash_attn_2_available()
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from headkv.snapkv_utils import init_headkv,init_reason_snapkv,  DynamicCacheSplitHeadFlatten

logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func


def _safe_decode_from_flatten_cache(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    bsz: int,
    num_heads: int,
    q_len: int,
    head_dim: int,
    hidden_size: int,
) -> torch.Tensor:
    query_flat = query_states[:, 0, :]
    key_flat = key_states[:, 0, :]
    value_flat = value_states[:, 0, :]

    outputs = []
    scale = math.sqrt(head_dim)
    for head_idx in range(num_heads):
        start = int(cu_seqlens_k[head_idx].item())
        end = int(cu_seqlens_k[head_idx + 1].item())
        k_head = key_flat[start:end]
        v_head = value_flat[start:end]
        q_head = query_flat[head_idx : head_idx + 1]

        attn_scores = torch.matmul(q_head, k_head.transpose(-1, -2)) / scale
        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_head.dtype)
        head_out = torch.matmul(attn_probs, v_head)
        outputs.append(head_out)

    out = torch.cat(outputs, dim=0)
    out = out.view(bsz, num_heads, q_len, head_dim).transpose(1, 2).reshape(bsz, q_len, hidden_size)
    return out


def _refresh_varlen_decode_metadata(kv_cluster):
    head_lens = kv_cluster.head_lens.to(dtype=torch.int32)
    device = head_lens.device

    kv_cluster.head_lens = head_lens
    kv_cluster.klen_sum = head_lens.sum(dtype=torch.int32)
    kv_cluster.max_seqlen_k = head_lens.max()

    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(head_lens, dim=0, dtype=torch.int32)[:-1],
        ],
        dim=0,
    )
    total_len = kv_cluster.klen_sum.reshape(1).to(dtype=torch.int32, device=device)
    kv_cluster.cu_klen = torch.cat([starts, total_len], dim=0)


def adaptive_LlamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,  # compat 4.45: new params
) -> Union[Tuple, BaseModelOutputWithPast]:
    import time

    call_start = time.perf_counter()
    timing = getattr(self, "_cake_timing", None)
    # re-init timing on prefill or first decode
    if timing is None or (input_ids is not None and input_ids.shape[1] > 1):
        timing = {
            "request_start": call_start,
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "decode_steps": 0,
            "ttft": None,
            "tpot": None,
            "latency": None,
        }
        self._cake_timing = timing

    is_prefill = False
    if use_cache is None:
        use_cache = self.config.use_cache
    if output_attentions is None:
        output_attentions = self.config.output_attentions
    if output_hidden_states is None:
        output_hidden_states = self.config.output_hidden_states
    if return_dict is None:
        return_dict = self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
        use_cache = False

    if use_cache:
        if past_key_values is None:
            past_key_values = DynamicCacheSplitHeadFlatten()
        elif not isinstance(past_key_values, DynamicCacheSplitHeadFlatten):
            if isinstance(past_key_values, Cache) and hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values.to_legacy_cache())
            else:
                past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_length,
            device=input_ids.device if input_ids is not None else inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if past_key_values is None:
        is_prefill = True
    else:
        is_prefill = cache_position.numel() > 1 or seq_length > 1

    if is_prefill:
        torch.cuda.synchronize()
        t = time.time()

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache

    if is_prefill:
        torch.cuda.synchronize()
        prefill_t = time.time() - t
        timing["prefill_time"] = prefill_t
        # TTFT recorded only on first prefill
        if timing["ttft"] is None:
            timing["ttft"] = time.perf_counter() - timing["request_start"]
        timing["latency"] = time.perf_counter() - timing["request_start"]
    else:
        # decode token
        decode_t = time.perf_counter() - call_start
        timing["decode_time"] += decode_t
        timing["decode_steps"] += 1
        if timing["ttft"] is None:
            timing["ttft"] = time.perf_counter() - timing["request_start"]
        timing["latency"] = time.perf_counter() - timing["request_start"]
        timing["tpot"] = timing["decode_time"] / max(timing["decode_steps"], 1)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def adaptive_llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,  # compat 4.45
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # compat 4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # NOTE: adakv
    init_headkv(self)
    if past_key_value is not None and not isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
        if isinstance(past_key_value, Cache) and hasattr(past_key_value, "to_legacy_cache"):
            past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value.to_legacy_cache())
        else:
            past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value)

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.45. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    tm = getattr(self.config, '_time_manager', None)
    if tm is not None:
        tm.record_qkv_proj_start(self.layer_idx)
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    if tm is not None:
        tm.record_qkv_proj_end(self.layer_idx)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    dropout_rate = self.attention_dropout if self.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    def _maybe_disable_unpad(mask: Optional[torch.Tensor], query_len: int) -> Optional[torch.Tensor]:
        if query_len <= 1 or mask is None:
            return None if query_len <= 1 else mask
        # FlashAttention's unpadding path expects a 2D padding mask [bsz, seq_len].
        # Any other mask shape (e.g., 4D causal masks) can explode memory in _upad_input.
        if mask.dim() != 2:
            return None
        if mask.dim() == 2 and torch.all(mask == 1):
            return None
        return mask

    # timing events (optional, only set by HeadKV time tests)
    te = getattr(self.config, '_timing_events', None)

    if past_key_value is None:
        if te is not None:
            te['pref_pure_start'][self.layer_idx].record()
        flash_attention_mask = _maybe_disable_unpad(attention_mask, q_len)
        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            flash_attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
            is_causal=getattr(self, "is_causal", True),
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        if te is not None:
            te['pref_pure_end'][self.layer_idx].record()
    else:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        is_prefill = q_len > 1
        if isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            has_layer_cache = (
                len(past_key_value.key_cache) > self.layer_idx and past_key_value.key_cache[self.layer_idx] is not None
            )
            is_prefill = is_prefill or (not has_layer_cache)
        if is_prefill:
            self.kv_seq_len = kv_seq_len
            if te is not None:
                te['pref_idx_start'][self.layer_idx].record()
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            if tm is not None:
                tm.record_write_cache_start(self.layer_idx)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            if tm is not None:
                tm.record_write_cache_end(self.layer_idx)
            if te is not None:
                te['pref_idx_end'][self.layer_idx].record()

            flash_attention_mask = _maybe_disable_unpad(attention_mask, q_len)
            if te is not None:
                te['pref_pure_start'][self.layer_idx].record()
            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                flash_attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
                is_causal=getattr(self, "is_causal", True),
            )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
            if te is not None:
                te['pref_pure_end'][self.layer_idx].record()

        else:
            self.kv_seq_len += q_len

            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            if tm is not None:
                tm.record_write_cache_start(self.layer_idx)
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if tm is not None:
                tm.record_write_cache_end(self.layer_idx)
            if tm is not None:
                tm.record_dec_retrieve_start(self.layer_idx)
            self.kv_cluster.head_lens = self.kv_cluster.head_lens + q_len
            _refresh_varlen_decode_metadata(self.kv_cluster)
            if tm is not None:
                tm.record_dec_retrieve_end(self.layer_idx)

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1, 1, self.head_dim)
            value_states = value_states.view(-1, 1, self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = (
                int(self.kv_cluster.max_seqlen_k.item())
                if torch.is_tensor(self.kv_cluster.max_seqlen_k)
                else int(self.kv_cluster.max_seqlen_k)
            )

            use_fast_decode = (
                _HEADKV_FLASH_ATTN_2_AVAILABLE
                and not getattr(self, "_disable_varlen_decode", False)
                and not getattr(self.config, "disable_varlen_decode", False)
            )
            if tm is not None:
                tm.record_dec_pure_start(self.layer_idx)
            if use_fast_decode:
                try:
                    attn_output = flash_attn_varlen_func(
                        query_states,
                        key_states,
                        value_states,
                        self.kv_cluster.cu_qlen,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        causal=True,
                    ).reshape(bsz, self.num_heads, q_len, self.head_dim)
                    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
                except RuntimeError:
                    self._disable_varlen_decode = True
                    attn_output = _safe_decode_from_flatten_cache(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        cu_seqlens_k=cu_seqlens_k,
                        bsz=bsz,
                        num_heads=self.num_heads,
                        q_len=q_len,
                        head_dim=self.head_dim,
                        hidden_size=self.hidden_size,
                    )
            else:
                attn_output = _safe_decode_from_flatten_cache(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    cu_seqlens_k=cu_seqlens_k,
                    bsz=bsz,
                    num_heads=self.num_heads,
                    q_len=q_len,
                    head_dim=self.head_dim,
                    hidden_size=self.hidden_size,
                )
            if tm is not None:
                tm.record_dec_pure_end(self.layer_idx)

    if tm is not None:
        tm.record_o_proj_start(self.layer_idx)
    attn_output = self.o_proj(attn_output)
    if tm is not None:
        tm.record_o_proj_end(self.layer_idx)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def reason_llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,  # compat 4.45
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # compat 4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # NOTE: reasonkv
    init_reason_snapkv(self)
    if past_key_value is not None and not isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
        if isinstance(past_key_value, Cache) and hasattr(past_key_value, "to_legacy_cache"):
            past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value.to_legacy_cache())
        else:
            past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value)

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.45. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    tm = getattr(self.config, '_time_manager', None)
    if tm is not None:
        tm.record_qkv_proj_start(self.layer_idx)
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    if tm is not None:
        tm.record_qkv_proj_end(self.layer_idx)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    dropout_rate = self.attention_dropout if self.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # timing events (optional, only set by HeadKV time tests)
    te = getattr(self.config, '_timing_events', None)

    if past_key_value is None:
        if te is not None:
            te['pref_pure_start'][self.layer_idx].record()
        flash_attention_mask = attention_mask if q_len > 1 else None
        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            flash_attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
            is_causal=getattr(self, "is_causal", True),
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        if te is not None:
            te['pref_pure_end'][self.layer_idx].record()
    else:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        is_prefill = q_len > 1
        if isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            has_layer_cache = (
                len(past_key_value.key_cache) > self.layer_idx and past_key_value.key_cache[self.layer_idx] is not None
            )
            is_prefill = is_prefill or (not has_layer_cache)
        if is_prefill:
            self.kv_seq_len = kv_seq_len
            if te is not None:
                te['pref_idx_start'][self.layer_idx].record()
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            if tm is not None:
                tm.record_write_cache_start(self.layer_idx)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            if tm is not None:
                tm.record_write_cache_end(self.layer_idx)
            if te is not None:
                te['pref_idx_end'][self.layer_idx].record()

            flash_attention_mask = attention_mask if q_len > 1 else None
            if te is not None:
                te['pref_pure_start'][self.layer_idx].record()
            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                flash_attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
                is_causal=getattr(self, "is_causal", True),
            )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
            if te is not None:
                te['pref_pure_end'][self.layer_idx].record()

        else:
            self.kv_seq_len += q_len

            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            if tm is not None:
                tm.record_write_cache_start(self.layer_idx)
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if tm is not None:
                tm.record_write_cache_end(self.layer_idx)
            if tm is not None:
                tm.record_dec_retrieve_start(self.layer_idx)
            self.kv_cluster.head_lens = self.kv_cluster.head_lens + q_len
            _refresh_varlen_decode_metadata(self.kv_cluster)
            if tm is not None:
                tm.record_dec_retrieve_end(self.layer_idx)

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1, 1, self.head_dim)
            value_states = value_states.view(-1, 1, self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = (
                int(self.kv_cluster.max_seqlen_k.item())
                if torch.is_tensor(self.kv_cluster.max_seqlen_k)
                else int(self.kv_cluster.max_seqlen_k)
            )

            use_fast_decode = (
                _HEADKV_FLASH_ATTN_2_AVAILABLE
                and not getattr(self, "_disable_varlen_decode", False)
                and not getattr(self.config, "disable_varlen_decode", False)
            )
            if tm is not None:
                tm.record_dec_pure_start(self.layer_idx)
            if use_fast_decode:
                try:
                    attn_output = flash_attn_varlen_func(
                        query_states,
                        key_states,
                        value_states,
                        self.kv_cluster.cu_qlen,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        causal=True,
                    ).reshape(bsz, self.num_heads, q_len, self.head_dim)
                    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
                except RuntimeError:
                    self._disable_varlen_decode = True
                    attn_output = _safe_decode_from_flatten_cache(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        cu_seqlens_k=cu_seqlens_k,
                        bsz=bsz,
                        num_heads=self.num_heads,
                        q_len=q_len,
                        head_dim=self.head_dim,
                        hidden_size=self.hidden_size,
                    )
            else:
                attn_output = _safe_decode_from_flatten_cache(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    cu_seqlens_k=cu_seqlens_k,
                    bsz=bsz,
                    num_heads=self.num_heads,
                    q_len=q_len,
                    head_dim=self.head_dim,
                    hidden_size=self.hidden_size,
                )
            if tm is not None:
                tm.record_dec_pure_end(self.layer_idx)

    if tm is not None:
        tm.record_o_proj_start(self.layer_idx)
    attn_output = self.o_proj(attn_output)
    if tm is not None:
        tm.record_o_proj_end(self.layer_idx)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def prepare_inputs_for_generation_llama(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    num_logits_to_keep=None,
    **kwargs,
):
    if past_key_values is not None:
        if inputs_embeds is not None:
            if cache_position is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
        elif cache_position is not None and input_ids.shape[1] != cache_position.shape[0]:
            input_ids = input_ids[:, cache_position]

    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    if inputs_embeds is not None and cache_position is not None and cache_position.numel() > 0 and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache if "use_cache" not in kwargs else kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    if num_logits_to_keep is not None:
        model_inputs["num_logits_to_keep"] = num_logits_to_keep
    return model_inputs
