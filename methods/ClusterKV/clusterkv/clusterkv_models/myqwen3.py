"""ClusterKV 效率后端的稠密 Qwen3 适配器。"""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .myqwen2 import Qwen2ForCausalLM


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    """复用 ClusterKV 的 Qwen 解码器和缓存路径，适配 Qwen3 注意力模块。"""

    config_class = Qwen3Config
