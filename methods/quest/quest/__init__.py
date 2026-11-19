"""Quest model exports; native kernel selection is handled by the loader."""

from importlib import import_module

__version__ = "0.0.1"
__all__ = ["LlamaForCausalLM", "Qwen2ForCausalLM"]


def __getattr__(name):
  modules = {
    "LlamaForCausalLM": "quest.models.llama",
    "Qwen2ForCausalLM": "quest.models.qwen2",
  }
  if name not in modules:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  value = getattr(import_module(modules[name]), name)
  globals()[name] = value
  return value
