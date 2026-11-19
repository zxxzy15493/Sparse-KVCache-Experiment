"""Equal-length batched inference using Quest's original native operator design."""

__all__ = ['NativeBatchLlamaForCausalLM']

def __getattr__(name):
    if name == 'NativeBatchLlamaForCausalLM':
        from .modeling_llama import NativeBatchLlamaForCausalLM
        return NativeBatchLlamaForCausalLM
    raise AttributeError(name)
