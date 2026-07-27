import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
    is_flash_attn_2_available,
)
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from headkv.snapkv_utils import init_headkv,init_reason_snapkv,  DynamicCacheSplitHeadFlatten

logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func


def _maybe_disable_unpad(mask: Optional[torch.Tensor], query_len: int) -> Optional[torch.Tensor]:
    if query_len <= 1 or mask is None:
        return None if query_len <= 1 else mask
    if mask.dim() != 2:
        return None
    if mask.dim() == 2 and torch.all(mask == 1):
        return None
    return mask

def adaptive_qwen2Model_forward(
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
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    import time
    call_start = time.perf_counter()
    timing = getattr(self, "_cake_timing", None)
    # re-init timing on prefill or first decode
    is_prefill = past_key_values is None
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
    if is_prefill:
        torch.cuda.synchronize()
        t = time.time()

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        if past_key_values is None:
            past_key_values = DynamicCacheSplitHeadFlatten()
        elif not isinstance(past_key_values, DynamicCacheSplitHeadFlatten):
            past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if cache_position is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        cache_position = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_length,
            dtype=torch.long,
            device=device,
        )

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # if self._use_flash_attention_2:
    if self.config._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds


    # decoder layers
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
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            if is_prefill:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    if is_prefill:
        torch.cuda.synchronize()
        prefill_t = time.time() - t
        timing["prefill_time"] = prefill_t
        if timing["ttft"] is None:
            timing["ttft"] = time.perf_counter() - timing["request_start"]
        timing["latency"] = time.perf_counter() - timing["request_start"]
    else:
        decode_t = time.perf_counter() - call_start
        timing["decode_time"] += decode_t
        timing["decode_steps"] += 1
        if timing["ttft"] is None:
            timing["ttft"] = time.perf_counter() - timing["request_start"]
        timing["latency"] = time.perf_counter() - timing["request_start"]
        timing["tpot"] = timing["decode_time"] / max(timing["decode_steps"], 1)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def adaptive_qwen2_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # NOTE: adakv
    init_headkv(self)
    if past_key_value is not None and not isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
        past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value)
    # Qwen2Attention attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.45. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    # ============================================================
    # ============================================================
    
    kv_seq_len = key_states.shape[-2]
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = self.rotary_emb(value_states, position_ids)
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    # ============================================================
    # ============================================================
    
    # [SnapKV] move to ahead
    if self.num_key_value_groups > 1:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        is_prefill = q_len > 1
        if isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            has_layer_cache = (
                len(past_key_value.key_cache) > self.layer_idx and past_key_value.key_cache[self.layer_idx] is not None
            )
            is_prefill = is_prefill or (not has_layer_cache)

        if is_prefill:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            query_states_t = query_states.transpose(1, 2)
            key_states_t = key_states.transpose(1, 2)
            value_states_t = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in the correct dtype just to be sure everything works as expected.
            # This might slowdown training & inference so it is recommended to not cast the LayerNorms
            # in fp32. (Qwen2RMSNorm handles it correctly)

            input_dtype = query_states_t.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states_t = query_states_t.to(target_dtype)
                key_states_t = key_states_t.to(target_dtype)
                value_states_t = value_states_t.to(target_dtype)

            flash_attention_mask = _maybe_disable_unpad(attention_mask, q_len)
            if hasattr(self, '_flash_attention_forward'):
                attn_output = self._flash_attention_forward(
                    query_states_t, key_states_t, value_states_t, flash_attention_mask, q_len, dropout=dropout_rate
                )
            else:
                attn_func = flash_attn_func if "flash_attn_func" in globals() else None
                if attn_func is None:
                    from flash_attn import flash_attn_func as _flash_attn_func
                    attn_func = _flash_attn_func
                attn_output = attn_func(
                    query_states_t, key_states_t, value_states_t,
                    dropout_p=dropout_rate,
                    causal=True,
                )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        else:
            self.kv_seq_len = getattr(self, "kv_seq_len", 0) + q_len

            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # NOTE: update meta data
            self.kv_cluster.klen_sum += self.num_heads
            self.kv_cluster.max_seqlen_k += 1
            self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
            self.kv_cluster.head_lens += 1
            if os.getenv("HEADKV_CALC_DECODE_METRICS", "0") == "1" and hasattr(self.kv_cluster, 'calculate_decode_metrics_only'):
                self.kv_cluster.calculate_decode_metrics_only(key_states, query_states, value_states)

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1,1,self.head_dim)
            value_states = value_states.view(-1,1,self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = self.kv_cluster.max_seqlen_k

            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=True,
            ).reshape(bsz, self.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(0, 1).reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def reason_qwen2_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # NOTE: reasonkv
    # Ensure config carries model path/name expected by ReasonSnapKVCluster
    # Try multiple fallbacks so Qwen2 behaves like LLaMA when loaded differently
    model_name = getattr(self.config, "_name_or_path", None) or getattr(self.config, "name_or_path", None) or getattr(self.config, "model_name", None)
    if not model_name:
        # fallback to introspecting module/class name
        try:
            model_name = getattr(self, "__class__").__module__ or ""
        except Exception:
            model_name = ""
    # set normalized value on config for downstream utils
    self.config._name_or_path = model_name
    # ensure head_choice exists for ReasonKV
    if not hasattr(self.config, "head_choice") or self.config.head_choice is None:
        self.config.head_choice = "reason"
    init_reason_snapkv(self)
    if past_key_value is not None and not isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
        past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value)
    # Qwen2Attention attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.45. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    # ============================================================
    # ============================================================
    
    kv_seq_len = key_states.shape[-2]
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = self.rotary_emb(value_states, position_ids)
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [SnapKV] move to ahead
    if self.num_key_value_groups > 1:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        is_prefill = q_len > 1
        if isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            has_layer_cache = (
                len(past_key_value.key_cache) > self.layer_idx and past_key_value.key_cache[self.layer_idx] is not None
            )
            is_prefill = is_prefill or (not has_layer_cache)

        if is_prefill:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            query_states_t = query_states.transpose(1, 2)
            key_states_t = key_states.transpose(1, 2)
            value_states_t = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in the correct dtype just to be sure everything works as expected.
            # This might slowdown training & inference so it is recommended to not cast the LayerNorms
            # in fp32. (Qwen2RMSNorm handles it correctly)

            input_dtype = query_states_t.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states_t = query_states_t.to(target_dtype)
                key_states_t = key_states_t.to(target_dtype)
                value_states_t = value_states_t.to(target_dtype)

            flash_attention_mask = _maybe_disable_unpad(attention_mask, q_len)
            if hasattr(self, '_flash_attention_forward'):
                attn_output = self._flash_attention_forward(
                    query_states_t, key_states_t, value_states_t, flash_attention_mask, q_len, dropout=dropout_rate
                )
            else:
                attn_func = flash_attn_func if "flash_attn_func" in globals() else None
                if attn_func is None:
                    from flash_attn import flash_attn_func as _flash_attn_func
                    attn_func = _flash_attn_func
                attn_output = attn_func(
                    query_states_t, key_states_t, value_states_t,
                    dropout_p=dropout_rate,
                    causal=True,
                )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        else:
            self.kv_seq_len = getattr(self, "kv_seq_len", 0) + q_len

            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # NOTE: update meta data
            self.kv_cluster.klen_sum += self.num_heads
            self.kv_cluster.max_seqlen_k += 1
            self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
            self.kv_cluster.head_lens += 1

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1,1,self.head_dim)
            value_states = value_states.view(-1,1,self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = self.kv_cluster.max_seqlen_k

            attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                                 cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True).reshape(
                bsz, self.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(0, 1).reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def prepare_inputs_for_generation_qwen2(
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
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if inputs_embeds is not None:
            if cache_position is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
        elif cache_position is not None and input_ids.shape[1] != cache_position.shape[0]:
            input_ids = input_ids[:, cache_position]

    if position_ids is None:
        position_ids = kwargs.get("position_ids", None)
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
