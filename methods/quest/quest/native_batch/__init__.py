"""Equal-length batched inference using Quest's original native operator design."""

__all__ = [
    'NativeBatchLlamaForCausalLM',
    'NativeBatchQwen2ForCausalLM',
    'NativeBatchQwen3ForCausalLM',
]

def __getattr__(name):
    if name == 'NativeBatchLlamaForCausalLM':
        from .modeling_llama import NativeBatchLlamaForCausalLM
        return NativeBatchLlamaForCausalLM
    if name == 'NativeBatchQwen2ForCausalLM':
        from .modeling_llama import NativeBatchQwen2ForCausalLM
        return NativeBatchQwen2ForCausalLM
    if name == 'NativeBatchQwen3ForCausalLM':
        from .modeling_llama import NativeBatchQwen3ForCausalLM
        return NativeBatchQwen3ForCausalLM
    raise AttributeError(name)
