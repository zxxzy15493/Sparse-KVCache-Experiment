import math
import os
import types
from typing import Optional, Tuple, Union, List, Dict, Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import ModelOutput
from .modeling_chatglm import (
    ChatGLMModel,
    ChatGLMForConditionalGeneration,
    GLMTransformer,
    GLMBlock,
    SelfAttention,
    apply_rotary_pos_emb,
)
from .baseline_compressor import *
from .retrieval_based.pq_search import *
from .retrieval_based.sparq import *

try:
    from flash_attn import flash_attn_func
except Exception:
    flash_attn_func = None


def layer2device(idx: int, layer_cnt: int) -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_in_use = len(visible.split(","))
    step = math.ceil(layer_cnt / gpu_in_use)
    return torch.device(f"cuda:{idx // step}")


def get_device(module: nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


def _expand_mqa_if_needed(attn: SelfAttention, key_layer: torch.Tensor, value_layer: torch.Tensor):
    """
    ChatGLM MQA path:
        key/value: [b, n_kv, s, h]
        query    : [b, n_q,  s, h]
    Expand kv to num_attention_heads when multi_query_attention=True.
    """
    if not attn.multi_query_attention:
        return key_layer, value_layer

    key_layer = key_layer.unsqueeze(2)
    key_layer = key_layer.expand(
        -1, -1, attn.num_attention_heads_per_partition // attn.num_multi_query_groups_per_partition, -1, -1
    )
    key_layer = key_layer.contiguous().view(
        key_layer.size()[:1] + (attn.num_attention_heads_per_partition,) + key_layer.size()[3:]
    )

    value_layer = value_layer.unsqueeze(2)
    value_layer = value_layer.expand(
        -1, -1, attn.num_attention_heads_per_partition // attn.num_multi_query_groups_per_partition, -1, -1
    )
    value_layer = value_layer.contiguous().view(
        value_layer.size()[:1] + (attn.num_attention_heads_per_partition,) + value_layer.size()[3:]
    )
    return key_layer, value_layer


def ChatGLMSelfAttentionPatch(attn: SelfAttention, config, layer_idx: int):
    """
    Patch ChatGLM SelfAttention.forward with a Llama-style monkey-patch workflow,
    while preserving ChatGLM's:
      - qkv projection layout
      - rotary cache / rope application
      - attention mask semantics
      - kv_cache interface
      - multi-query attention expansion logic
    """
    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        bsz, q_len, _ = hidden_states.size()

        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            query_layer, key_layer, value_layer = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                bsz, q_len, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head
            )
            key_layer = key_layer.view(
                bsz, q_len, self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head
            )
            value_layer = value_layer.view(
                bsz, q_len, self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head
            )
        else:
            mixed_x_layer = mixed_x_layer.view(
                bsz, q_len, self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head
            )
            query_layer, key_layer, value_layer = torch.split(
                mixed_x_layer, self.hidden_size_per_attention_head, dim=-1
            )

        query_layer = query_layer.transpose(1, 2).contiguous()
        key_layer = key_layer.transpose(1, 2).contiguous()
        value_layer = value_layer.transpose(1, 2).contiguous()

        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        first_time = (q_len > 1)

        # -------- branch 1: original / full-cache methods --------
        if self.compressor in ["original", "no_drop_lb", "no_drop_lb_32", "no_drop_lb_topp", "no_drop_lb_topp32", "h2o"]:
            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                key_layer = torch.cat((cache_k, key_layer), dim=2)
                value_layer = torch.cat((cache_v, value_layer), dim=2)

            present = (key_layer, value_layer) if use_cache else None

            expanded_key, expanded_value = _expand_mqa_if_needed(self, key_layer, value_layer)

            if self.compressor == "original":
                attn_output = flash_attn_func(
                        query_layer.transpose(1, 2),
                        expanded_key.transpose(1, 2),
                        expanded_value.transpose(1, 2),
                        causal=True,
                    ).transpose(1, 2)
            else:
                kv = (expanded_key, expanded_value)
                if self.use_flash_attn and first_time:
                    attn_output = flash_attn_func(
                        query_layer.transpose(1, 2),
                        expanded_key.transpose(1, 2),
                        expanded_value.transpose(1, 2),
                        causal=True,
                    ).transpose(1, 2)
                    self.kvcache_quantizer.apply(kv,layer_idx=attn.idx, query_states=query_layer)
                else:
                    attn_weights = torch.matmul(query_layer, expanded_key.transpose(2, 3))
                    attn_weights = attn_weights / math.sqrt(self.hidden_size_per_attention_head)
                    attn_weights = self.kvcache_quantizer.restore(
                        attn_weights, getattr(self, "num_key_value_groups", 1),
                        query_states=query_layer,
                        key_states=key_layer,
                        layer_idx=attn.idx,
                    ).to(query_layer.dtype)
                    attn_output = torch.matmul(attn_weights, expanded_value)

        # -------- branch 2: retrieval compressors --------
        elif self.compressor in ["pq_search", "sparq_f"]:
            present = kv_cache if use_cache else None

            if first_time:
                attn_output, _ = self.kvcache_quantizer.prefill_attn(
                    query_layer, (key_layer, value_layer)
                )

                # To keep a tiny seed cache, store only the first token; or skip entirely if not needed
                # if use_cache:
                #     present = (
                #         key_layer[..., :1, :].clone(),
                #         value_layer[..., :1, :].clone(),
                #     )
            else:
                cur_key, cur_value = _expand_mqa_if_needed(self, key_layer, value_layer)
                attn_output = self.kvcache_quantizer.decoding_attn(
                    getattr(self, "num_key_value_groups", 1),
                    query_layer,
                    cur_key,
                    cur_value,
                ).to(query_layer.dtype)

        else:
            raise ValueError(f"Unknown compressor: {self.compressor}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.projection_size)
        output = self.dense(attn_output)
        return output, present

    attn.forward = types.MethodType(forward, attn)
    attn.use_flash_attn = True
    attn.fwd_cnt = 0
    attn.seq_cnt = -1
    attn.idx = layer_idx
    attn.score_func = getattr(config, "score_func", "sum")
    attn.compressor = getattr(config, "compressor", "original")
    attn.num_key_value_groups = max(
        1,
        config.num_attention_heads // getattr(config, "multi_query_group_num", config.num_attention_heads),
    )

    if attn.compressor == "h2o":
        attn.kvcache_quantizer = KVCacheH2OOfficial(
            config.compress_ratio,
            config.important_ratio,
            config.recent_ratio,
            config.sink_size,
        )
    elif attn.compressor == "no_drop_lb":
        attn.kvcache_quantizer = fullKVLimitBasedCompressor(
            config.compress_ratio,
            config.important_ratio,
            config.recent_ratio,
            getattr(config, "gqa", True),
            config.sink_size,
            fixbudget=getattr(config, "fixbudget", True),
            budget=getattr(config, "budget", 0),
            recent_size=getattr(config, "recent_size", 32),
        )
    elif attn.compressor == "no_drop_lb_32":
        attn.kvcache_quantizer = fullKVLimitBasedCompressor32(
            config.compress_ratio,
            config.important_ratio,
            config.recent_ratio,
            getattr(config, "gqa", True),
            config.sink_size,
            fixbudget=getattr(config, "fixbudget", True),
            budget=getattr(config, "budget", 0),
            recent_size=getattr(config, "recent_size", 32),
        )
    elif attn.compressor == "no_drop_lb_topp":
        attn.kvcache_quantizer = fullKVLimitBasedCompressorTOPP(
            config.fixthreshold,
            getattr(config, "gqa", True),
        )
    elif attn.compressor == "no_drop_lb_topp32":
        attn.kvcache_quantizer = fullKVLimitBasedCompressorTOPP32(
            config.fixthreshold,
            getattr(config, "gqa", True),
        )
    elif attn.compressor == "pq_search":
        attn.kvcache_quantizer = PqBasedSearchCompressor(
            config.compress_ratio,
            config.recent_ratio,
            config.n_subvec_per_head,
            config.n_subbits,
            getattr(config, "gqa", True),
            config.sink_size,
            fixbudget=getattr(config, "fixbudget", False),
            budget=getattr(config, "budget", 0),
            recent_size=getattr(config, "recent_size", 32),
            layer_idx=layer_idx,
            cur_device=layer2device(layer_idx, config.num_layers),
            max_iter=config.max_iter,
            kv_head=getattr(config, "multi_query_group_num", config.num_attention_heads),
            dim=config.hidden_size // config.num_attention_heads,
            num_layer_cnt=config.num_layers,
        )
    elif attn.compressor == "sparq_f":
        attn.kvcache_quantizer = SparQCompressor(
            config.compress_ratio,
            config.recent_ratio,
            config.sink_size,
            getattr(config, "gqa", True),
            r=config.topr,
            idx=layer_idx,
            model_config=config,
            layer_idx=layer_idx,
            cur_device=layer2device(layer_idx, config.num_layers),
            kv_head=getattr(config, "multi_query_group_num", config.num_attention_heads),
            dim=config.hidden_size // getattr(config, "multi_query_group_num", 1),
        )
    elif attn.compressor == "original":
        pass
    else:
        raise ValueError(f"Unknown compressor: {attn.compressor}")

    return attn


# def ChatGLMBlockPatch(layer: GLMBlock, config, layer_idx: int):
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.BoolTensor],
#         rotary_pos_emb: Optional[torch.Tensor],
#         kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#         use_cache: bool = True,
#     ):
#         # Keep ChatGLM residual + LN order exactly as original
#         layernorm_output = self.input_layernorm(hidden_states)

#         attention_output, present = self.self_attention(
#             layernorm_output,
#             attention_mask,
#             rotary_pos_emb,
#             kv_cache=kv_cache,
#             use_cache=use_cache,
#         )

#         residual = layernorm_output if self.apply_residual_connection_post_layernorm else hidden_states
#         hidden_states = residual + torch.nn.functional.dropout(
#             attention_output, p=self.hidden_dropout, training=self.training
#         )

#         layernorm_output = self.post_attention_layernorm(hidden_states)
#         mlp_output = self.mlp(layernorm_output)

#         residual = layernorm_output if self.apply_residual_connection_post_layernorm else hidden_states
#         output = residual + torch.nn.functional.dropout(
#             mlp_output, p=self.hidden_dropout, training=self.training
#         )
#         return output, present

#     layer.forward = types.MethodType(forward, layer)
#     layer.device = layer2device(layer_idx, config.num_layers)
#     ChatGLMSelfAttentionPatch(layer.self_attention, config, layer_idx)
#     return layer
def ChatGLMBlockPatch(layer, config, layer_idx):
    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # Align with second patch: save residual first
        residual = hidden_states.clone()
        batch, seq_len, embed_dim = hidden_states.shape

        # 1) Align with second patch: chunk input_layernorm by 32000
        for start_idx in range(0, seq_len, 32000):
            end_idx = min(seq_len, start_idx + 32000)
            hidden_states[:, start_idx:end_idx, :] = self.input_layernorm(
                hidden_states[:, start_idx:end_idx, :]
            )

        # 2) Attention still uses ChatGLM's self_attention patch
        attn_output, present = self.self_attention(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        # 3) Align with second patch: inplace residual add after attention
        hidden_states = attn_output
        hidden_states.add_(residual)
        del residual

        # 4) Align with second patch: post-attn LN + MLP chunked
        n_chunks = max(1, math.ceil(seq_len / 32000))
        avg_chunk_size = math.ceil(seq_len / n_chunks)

        for start_idx in range(0, seq_len, avg_chunk_size):
            end_idx = min(seq_len, start_idx + avg_chunk_size)

            part_hidden_states = hidden_states[:, start_idx:end_idx, :].clone()
            part_hidden_states = self.post_attention_layernorm(part_hidden_states)
            part_hidden_states = self.mlp(part_hidden_states)
            hidden_states[:, start_idx:end_idx, :] += part_hidden_states

        return hidden_states, present

    layer.forward = types.MethodType(forward, layer)
    layer.device = layer2device(layer_idx, config.num_layers)
    ChatGLMSelfAttentionPatch(layer.self_attention, config, layer_idx)
    return layer

def ChatGLMTransformerPatch(transformer: GLMTransformer, config):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.BoolTensor],
        rotary_pos_emb: Optional[torch.Tensor],
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):
        if kv_caches is None:
            kv_caches = [None for _ in range(self.num_layers)]

        all_hidden_states = () if output_hidden_states else None
        presents = [] if use_cache else None
        all_self_attentions = None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if hidden_states.device != layer.device:
                hidden_states = hidden_states.to(layer.device)
            if rotary_pos_emb is not None and rotary_pos_emb.device != layer.device:
                rotary_pos_emb = rotary_pos_emb.to(layer.device)
            if attention_mask is not None and attention_mask.device != layer.device:
                attention_mask = attention_mask.to(layer.device)

            hidden_states, present = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=kv_caches[idx],
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        if self.post_layer_norm:
            if hidden_states.device != get_device(self.final_layernorm):
                hidden_states = hidden_states.to(get_device(self.final_layernorm))
            hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        presents = tuple(presents) if use_cache else None
        return hidden_states, presents, all_hidden_states, all_self_attentions

    transformer.forward = types.MethodType(forward, transformer)

    for i in range(config.num_layers):
        dev = layer2device(i, config.num_layers)
        transformer.layers[i] = ChatGLMBlockPatch(transformer.layers[i].to(dev), config, i)

    if transformer.post_layer_norm:
        transformer.final_layernorm = transformer.final_layernorm.to(
            layer2device(config.num_layers - 1, config.num_layers)
        )

    transformer.gradient_checkpointing = False
    return transformer


def ChatGLMModelPatch(model: ChatGLMModel, config):
    def forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if full_attention_mask is None:
            if seq_length == 1 and past_key_values:
                attention_mask = None
            elif (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
            else:
                attention_mask = None

        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=list(past_key_values) if past_key_values is not None else None,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    model.forward = types.MethodType(forward, model)

    # Pipe-style placement
    model.embedding = model.embedding.to(torch.device("cuda:0"))
    model.encoder = ChatGLMTransformerPatch(model.encoder, config)
    model.output_layer = model.output_layer.to(layer2device(config.num_layers - 1, config.num_layers))
    return model


def ChatGLMForConditionalGenerationPatch(model: ChatGLMForConditionalGeneration, config):
    """
    Patch entry point. Keeps ChatGLM public API, but changes internal execution
    into patch-injected workflow like the Llama example.
    """
    model.transformer = ChatGLMModelPatch(model.transformer, config)
    model.max_sequence_length = config.max_length
    model.gradient_checkpointing = False
    model.post_init()
    return model


class VQGlmForCausalLM(ChatGLMForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.layer_num = config.num_layers
        self.kv_head_cnt = config.multi_query_group_num

        self._device = torch.device("cuda:0")
        self.fwd_cnt = 0
        self.gen_seq_cnt = 0
        self.prefill_len = 0

        self.gradient_checkpointing = False

    def patch(self, config):
        ChatGLMForConditionalGenerationPatch(self, config)
        self.transformer.output_layer = self.transformer.output_layer.to(
            layer2device(config.num_layers - 1, config.num_layers)
        )
        self.post_init()

    
