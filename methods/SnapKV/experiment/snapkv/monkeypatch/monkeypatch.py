from importlib.metadata import version
import warnings
import transformers
from snapkv.monkeypatch.llama_hijack_4_45 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_45
from snapkv.monkeypatch.qwen_hijack_4_45 import qwen2_flash_attn2_forward as qwen2_flash_attn2_forward_4_45
from snapkv.monkeypatch.glm_hijack_4_45 import enable_snapkv_glm_attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama():
    transformers_version = check_version()
    version_list = ['4.45', '4.46']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    llama_modeling = transformers.models.llama.modeling_llama
    attention_cls = getattr(llama_modeling, "LlamaFlashAttention2", llama_modeling.LlamaAttention)
    attention_cls.forward = llama_flash_attn2_forward_4_45

def replace_qwen():
    transformers_version = check_version()
    version_list = ['4.45', '4.46']
    warning_flag = True
    for v in version_list:
        if v in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    qwen2_modeling = transformers.models.qwen2.modeling_qwen2
    attention_cls = getattr(qwen2_modeling, "Qwen2FlashAttention2", qwen2_modeling.Qwen2Attention)
    attention_cls.forward = qwen2_flash_attn2_forward_4_45

def replace_qwen3():
    """Patch Qwen3 attention with SnapKV's compatible Qwen attention path."""
    Qwen3Attention.forward = qwen2_flash_attn2_forward_4_45
def replace_glm(model):
    transformers_version = check_version()
    version_list = ['4.45', '4.46']
    warning_flag = True
    for v in version_list:
        if v in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    enable_snapkv_glm_attention(model)
