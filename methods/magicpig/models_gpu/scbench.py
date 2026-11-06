"""SCBench shared-prefix sessions for the MagicPIG HF GPU backend."""

import torch

from benchmarks.scbench.external import ExternalSession
from benchmarks.scbench.state import clone_state


class MagicPigGPUSession(ExternalSession):
    """Retain native HF KV and GPU LSH sampling across SCBench turns."""

    def __init__(self, args, model, tokenizer):
        if model.batch_size != 1:
            raise ValueError("SCBench evaluates serial samples with batch size one")
        super().__init__(args, model, tokenizer)
        eos = model.eos_token_ids
        self.stop_tokens.update([eos] if isinstance(eos, int) else (eos or []))

    def clear_model_state(self):
        # Counters describe executed work, so KV reset/rollback preserves them.
        self.model.clear()

    def _forward_native(self, ids, position):
        positions = torch.arange(position, position + ids.shape[1], device=self.device)
        output = self.model.model(
            input_ids=ids, past_key_values=self.model.past_key_values,
            position_ids=positions.unsqueeze(0), cache_position=positions,
            use_cache=True, return_dict=True, logits_to_keep=1,
        )
        self.model.past_key_values = output.past_key_values
        return output.logits[:, -1, :]

    def prefill(self, ids):
        from transformers import DynamicCache

        if self.position or self.model.past_key_values is not None:
            raise ValueError("Reset the MagicPIG GPU session before initial prefill")
        state = self.model.attention_server
        state.prompt_mask = torch.ones_like(ids, dtype=torch.bool)
        state.is_prefill = True
        try:
            cache = DynamicCache(config=self.model.config)
            if any(layer.is_sliding for layer in cache.layers):
                raise ValueError("MagicPIG GPU SCBench requires full-context Qwen KV caches")
            self.model.past_key_values = cache
            chunk = self.args.prefill_chunk_size or self.model.chunk_size
            for start in range(0, ids.shape[1], chunk):
                logits = self._forward_native(ids[:, start:start + chunk], start)
            # Initialize read-only context hashes before any request checkpoint,
            # so all branches reuse the same matrices, keys and inclusion rule.
            if state.K:
                for layer in self.model.model.model.layers:
                    attention = layer.self_attn
                    if attention.layer_idx % 16:
                        kv = cache.layers[attention.layer_idx]
                        state.prepare_layer(attention, kv.keys, kv.values)
            return logits
        except Exception:
            self.model.clear()
            raise
        finally:
            state.is_prefill = False

    def decode(self, ids):
        if self.model.past_key_values is None:
            raise ValueError("MagicPIG GPU decode requires initial prefill")
        return self._forward_native(ids, self.position)

    def checkpoint(self):
        # Full-context DynamicCache appends KV. Restoring a prefix length avoids
        # cloning the long context and read-only hash tables for every question.
        # Dynamic RoPE can update small buffers; save those independently.
        rotary = self.model.model.model.rotary_emb
        fields = {name: getattr(rotary, name) for name in (
            "inv_freq", "original_inv_freq", "max_seq_len_cached", "attention_scaling",
        ) if hasattr(rotary, name)}
        return self.model.past_key_values, self.position, clone_state(fields)

    def restore(self, checkpoint):
        cache, position, rotary = checkpoint
        if cache is None:
            self.model.clear()
        else:
            if cache is not self.model.past_key_values:
                raise ValueError("MagicPIG GPU checkpoint belongs to a different session")
            if any(layer.get_seq_length() < position for layer in cache.layers):
                raise ValueError("MagicPIG GPU checkpoint prefix has already been discarded")
            cache.crop(position)
        self.position = position
        for name, value in rotary.items():
            setattr(self.model.model.model.rotary_emb, name, clone_state(value))
