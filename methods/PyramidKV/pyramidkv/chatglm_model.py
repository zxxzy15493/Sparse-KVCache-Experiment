import sys
import types
import torch

from .pyramidkv_utils import init_pyramidkv

__all__ = ["enable_pyramidkv_glm_attention", "chatglm_transformer_forward_no_cat"]


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


def glm_pyramidkv_attention_forward(
	self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
	if not hasattr(self, "layer_idx"):
		self.layer_idx = max(getattr(self, "layer_number", 1) - 1, 0)
	init_pyramidkv(
		self,
		num_hidden_layers=getattr(self.config, "num_hidden_layers", getattr(self.config, "num_layers", 32)),
	)

	modeling_chatglm = sys.modules[self.__class__.__module__]
	split_tensor_along_last_dim = modeling_chatglm.split_tensor_along_last_dim
	apply_rotary_pos_emb = modeling_chatglm.apply_rotary_pos_emb

	mixed_x_layer = self.query_key_value(hidden_states)

	if self.multi_query_attention:
		(query_layer, key_layer, value_layer) = mixed_x_layer.split(
			[
				self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
				self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
				self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
			],
			dim=-1,
		)
		query_layer = query_layer.view(
			query_layer.size()[:-1]
			+ (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
		)
		key_layer = key_layer.view(
			key_layer.size()[:-1]
			+ (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
		)
		value_layer = value_layer.view(
			value_layer.size()[:-1]
			+ (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
		)
	else:
		new_tensor_shape = mixed_x_layer.size()[:-1] + (
			self.num_attention_heads_per_partition,
			3 * self.hidden_size_per_attention_head,
		)
		mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
		(query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

	query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

	if rotary_pos_emb is not None:
		query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
		key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

	key_layer, value_layer = _expand_multi_query(self, key_layer, value_layer)

	kv_cache = _normalize_kv_cache(kv_cache)
	if kv_cache is not None:
		cache_k, cache_v = kv_cache
		if isinstance(kv_cache, tuple) and len(cache_k.shape) == 5:
			cache_k, cache_v = cache_k[0], cache_v[0]
		# Align dimensions before cat (handle 3D/4D mismatch)
		while cache_k.dim() < key_layer.dim():
			cache_k = cache_k.unsqueeze(0)
			cache_v = cache_v.unsqueeze(0)
		key_layer = torch.cat((cache_k, key_layer), dim=2)
		value_layer = torch.cat((cache_v, value_layer), dim=2)

	cache_key_layer = key_layer
	cache_value_layer = value_layer
	if kv_cache is None and hasattr(self, "kv_cluster"):
		# Prefill compression for cache only.
		cache_key_layer, cache_value_layer = self.kv_cluster.update_kv(
			key_layer, query_layer, value_layer, attention_mask, 1
		)

	if use_cache:
		if kv_cache is None:
			kv_cache = torch.cat(
				(
					cache_key_layer.unsqueeze(0).unsqueeze(0),
					cache_value_layer.unsqueeze(0).unsqueeze(0),
				),
				dim=1,
			)
		else:
			kv_cache = (cache_key_layer, cache_value_layer)
	else:
		kv_cache = None

	context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
	output = self.dense(context_layer)

	return output, kv_cache


def enable_pyramidkv_glm_attention(model):
	if not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
		return

	for layer_idx, layer in enumerate(model.transformer.encoder.layers):
		attn = layer.self_attention
		attn.config = model.config
		attn.layer_idx = layer_idx
		attn.forward = types.MethodType(glm_pyramidkv_attention_forward, attn)


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
