__all__ = [
    'LlamaConfig',
    'LlamaForCausalLM',
]


def __getattr__(name):
    if name in __all__:
        from clusterkv.clusterkv_models.llama import LlamaConfig, LlamaForCausalLM

        values = {
            "LlamaConfig": LlamaConfig,
            "LlamaForCausalLM": LlamaForCausalLM,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
