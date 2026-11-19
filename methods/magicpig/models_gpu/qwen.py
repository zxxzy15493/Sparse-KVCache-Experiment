"""Qwen entry point for the MagicPIG GPU backend."""

from .model import MagicPigGPUModel


QWEN25_7B_1M = "Qwen/Qwen2.5-7B-Instruct-1M"


class Qwen2Model(MagicPigGPUModel):
    default_model_name = QWEN25_7B_1M
    supported_model_types = ("qwen2", "qwen3")
