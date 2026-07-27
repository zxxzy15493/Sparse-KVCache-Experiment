import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
    Qwen2FlashAttention2,
    _flash_attention_forward,
)
from transformers.utils import (
    logging,
)

from .pyramidkv_utils import init_pyramidkv

logger = logging.get_logger(__name__)


def _cuda_mem_snapshot():
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    free, total = torch.cuda.mem_get_info(device)
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    return f"dev={device} allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB free={free_gb:.2f}GiB total={total_gb:.2f}GiB"


def _register_mlp_debug_hooks(model):
    if getattr(model, "_mlp_debug_hooks_installed", False):
        return

    hook_handles = []

    for layer_idx, decoder_layer in enumerate(model.layers):
        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is None:
            continue

        def _make_hook(idx):
            def _hook(module, inputs):
                if getattr(model.config, "debug_mask_logits", False):
                    hidden_state = inputs[0]
                    print(
                        f"[DEBUG][MLP_IN] layer={idx} shape={tuple(hidden_state.shape)} dtype={hidden_state.dtype} "
                        f"device={hidden_state.device} {_cuda_mem_snapshot()}"
                    )
            return _hook

        hook_handles.append(mlp.register_forward_pre_hook(_make_hook(layer_idx)))

    model._mlp_debug_hook_handles = hook_handles
    model._mlp_debug_hooks_installed = True


def fixed_qwen2Model_forward(
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
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

    return_legacy_cache = False
    if use_cache:
        if past_key_values is None:
            past_key_values = DynamicCache()
        elif not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_length,
            dtype=torch.long,
            device=device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if getattr(self.config, "debug_mask_logits", False):
        _register_mlp_debug_hooks(self)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
    if getattr(self.config, "debug_mask_logits", False) and not getattr(self, "_debug_mask_printed", False):
        print(
            f"[DEBUG][CAUSAL_MASK] attention_mask_shape={None if attention_mask is None else tuple(attention_mask.shape)} "
            f"causal_mask_shape={None if causal_mask is None else tuple(causal_mask.shape)} "
            f"causal_mask_dtype={None if causal_mask is None else causal_mask.dtype}"
        )
        self._debug_mask_printed = True
    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for i, decoder_layer in enumerate(self.layers):
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
            before = torch.cuda.memory_allocated()
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
            after = torch.cuda.memory_allocated()
            # print(f"[MEM] layer {i}: {before/1e9:.2f}GiB -> {after/1e9:.2f}GiB delta={(after-before)/1e9:.2f}GiB peak={torch.cuda.max_memory_allocated()/1e9:.2f}GiB")

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        torch.cuda.empty_cache()

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if return_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_flash_attn2_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    import time
    call_start = time.perf_counter()
    bsz, q_len, _ = hidden_states.size()
    timing = getattr(self, "_cake_timing", None)
    if timing is None or q_len > 1:
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

    bsz, q_len, _ = hidden_states.size()
    mem_before = torch.cuda.memory_allocated()
    is_prefill = past_key_value is None or q_len > 1

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    mem_qkv = torch.cuda.memory_allocated()

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. if you are using {self.__class__.__name__} "
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

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    mem_rope = torch.cuda.memory_allocated()

    # KV cache stores unrepeated (4-head) tensors to save 8x memory
    cache_kwargs = {"sin": sin, "cos": cos}
    if past_key_value is not None and key_states.shape[-2] == kv_seq_len:
        self.kv_seq_len = kv_seq_len
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(
            key_states,
            query_states,
            value_states,
            attention_mask,
            self.num_key_value_groups,
        )
        mem_compress = torch.cuda.memory_allocated()
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        mem_cache_update = torch.cuda.memory_allocated()
    elif past_key_value is not None:
        self.kv_seq_len += q_len
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        mem_compress = torch.cuda.memory_allocated()
        mem_cache_update = torch.cuda.memory_allocated()
    else:
        mem_compress = torch.cuda.memory_allocated()
        mem_cache_update = torch.cuda.memory_allocated()

    # Only repeat K/V heads right before flash attention, not before caching
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    mem_repeat = torch.cuda.memory_allocated()

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    mem_before_attn = torch.cuda.memory_allocated()

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

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        is_causal=getattr(self, "is_causal", True),
        dropout=dropout_rate,
        position_ids=None,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
    )
    mem_after_attn = torch.cuda.memory_allocated()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    mem_after_o = torch.cuda.memory_allocated()

    # if not is_prefill and q_len == 1:
    #     print(f"[ATTN-DETAIL][layer={self.layer_idx}] "
    #           f"proj={mem_qkv-mem_before:.1f}GiB "
    #           f"rope={mem_rope-mem_qkv:.1f}GiB "
    #           f"repeat={mem_repeat-mem_rope:.1f}GiB "
    #           f"compress={mem_compress-mem_repeat:.1f}GiB "
    #           f"cache_update={mem_cache_update-mem_compress:.1f}GiB "
    #           f"before_attn={mem_before_attn-mem_cache_update:.1f}GiB "
    #           f"flash_attn={mem_after_attn-mem_before_attn:.1f}GiB "
    #           f"output={mem_after_o-mem_after_attn:.1f}GiB "
    #           f"total={mem_after_o-mem_before:.1f}GiB"
    #     )

    if not output_attentions:
        attn_weights = None

    # efficiency stats
    if is_prefill:
        timing["prefill_time"] = time.perf_counter() - call_start
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

    return attn_output, attn_weights, past_key_value


def prepare_inputs_for_generation_qwen2(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    num_logits_to_keep = kwargs.get("num_logits_to_keep", None)
    if past_key_values is not None and past_key_values.get_seq_length() == 0:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            # Use kv_seq_len (original prefill length) instead of get_seq_length() (compressed length)
            # to properly slice input_ids
            past_length = self.model.layers[0].self_attn.kv_seq_len
            cache_length = past_key_values.get_seq_length()
            max_cache_length = past_key_values.get_max_length() if hasattr(past_key_values, "get_max_length") else None
        else:
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    if num_logits_to_keep is not None:
        model_inputs["num_logits_to_keep"] = num_logits_to_keep
    return model_inputs
