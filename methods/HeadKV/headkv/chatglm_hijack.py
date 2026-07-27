import torch

from headkv.snapkv_utils import init_snapkv, init_pyramidkv


def _apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


def _expand_multi_query(self, key_layer, value_layer):
    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1,
            -1,
            self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
            -1,
            -1,
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
        )

        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1,
            -1,
            self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
            -1,
            -1,
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
        )
    return key_layer, value_layer


def _normalize_kv_cache(kv_cache):
    if kv_cache is None:
        return None
    if isinstance(kv_cache, tuple) and len(kv_cache) == 2:
        return kv_cache
    if torch.is_tensor(kv_cache):
        if kv_cache.dim() == 6 and kv_cache.size(0) == 1 and kv_cache.size(1) == 2:
            cache_k = kv_cache[:, 0].squeeze(0)
            cache_v = kv_cache[:, 1].squeeze(0)
            return cache_k, cache_v
        if kv_cache.dim() == 5 and kv_cache.size(0) == 2:
            return kv_cache[0], kv_cache[1]
    raise ValueError("Unsupported ChatGLM kv_cache format")


def _chatglm_headkv_forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True, mode="snap"):
    if not hasattr(self, "layer_idx"):
        self.layer_idx = max(getattr(self, "layer_number", 1) - 1, 0)

    if mode == "pyramid":
        init_pyramidkv(self)
    else:
        init_snapkv(self)

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
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_shape)
        query_layer, key_layer, value_layer = torch.split(
            mixed_x_layer, self.hidden_size_per_attention_head, dim=-1
        )

    query_layer, key_layer, value_layer = [x.transpose(1, 2) for x in (query_layer, key_layer, value_layer)]

    if rotary_pos_emb is not None:
        query_layer = _apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = _apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    key_layer, value_layer = _expand_multi_query(self, key_layer, value_layer)

    kv_cache = _normalize_kv_cache(kv_cache)
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)

    if kv_cache is None and hasattr(self, "kv_cluster"):
        key_layer, value_layer = self.kv_cluster.update_kv(key_layer, query_layer, value_layer)

    if use_cache:
        kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
    output = self.dense(context_layer)
    return output, kv_cache


def fixed_chatglm_flash_attn2_forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
    return _chatglm_headkv_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=kv_cache,
        use_cache=use_cache,
        mode="snap",
    )


def pyramidkv_chatglm_flash_attn2_forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
    return _chatglm_headkv_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=kv_cache,
        use_cache=use_cache,
        mode="pyramid",
    )


def adaptive_chatglm_flash_attn2_forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
    return _chatglm_headkv_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=kv_cache,
        use_cache=use_cache,
        mode="snap",
    )


def reason_chatglm_flash_attn2_forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
    return _chatglm_headkv_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=kv_cache,
        use_cache=use_cache,
        mode="snap",
    )


def chatglm_transformer_forward_no_cat(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_caches=None,
    use_cache: bool = True,
    output_hidden_states: bool = False,
):
    if not kv_caches:
        kv_caches = [None for _ in range(self.num_layers)]

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    presents = () if use_cache else None
    all_self_attentions = None
    all_hidden_states = () if output_hidden_states else None

    for index in range(self.num_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer = self._get_layer(index)
        if self.gradient_checkpointing and self.training:
            layer_ret = torch.utils.checkpoint.checkpoint(
                layer,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_caches[index],
                use_cache,
                use_reentrant=False,
            )
        else:
            layer_ret = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=kv_caches[index],
                use_cache=use_cache,
            )

        hidden_states, kv_cache = layer_ret
        if use_cache:
            presents = presents + (_normalize_kv_cache(kv_cache),)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states, presents, all_hidden_states, all_self_attentions