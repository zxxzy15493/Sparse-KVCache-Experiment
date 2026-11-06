from importlib.metadata import version
from importlib import import_module
import warnings
import transformers
from snapkv.monkeypatch.llama_hijack import llama_flash_attn2_forward
from snapkv.monkeypatch.qwen_hijack import qwen2_flash_attn2_forward
from snapkv.monkeypatch.glm_hijack import enable_snapkv_glm_attention

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama():
    transformers_version = check_version()
    version_list = ['4.57']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    llama_modeling = transformers.models.llama.modeling_llama
    attention_cls = getattr(llama_modeling, "LlamaFlashAttention2", llama_modeling.LlamaAttention)
    attention_cls.forward = llama_flash_attn2_forward

def replace_qwen():
    transformers_version = check_version()
    version_list = ['4.45', '4.46', '4.57']
    warning_flag = True
    for v in version_list:
        if v in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    # Retain the legacy patch only when using an older Transformers Qwen2 API.
    qwen2_attention = transformers.models.qwen2.modeling_qwen2
    if hasattr(qwen2_attention, "Qwen2FlashAttention2"):
        qwen2_attention.Qwen2FlashAttention2.forward = qwen2_flash_attn2_forward
    else:
        qwen2_attention.Qwen2Attention.forward = qwen2_flash_attn2_forward

def replace_qwen3():
    """Patch Qwen3 attention with SnapKV's compatible Qwen attention path."""
    try:
        qwen3_attention = import_module("transformers.models.qwen3.modeling_qwen3")
    except ModuleNotFoundError as exc:
        raise RuntimeError("The installed Transformers package does not provide Qwen3Attention.") from exc

    from snapkv.monkeypatch.qwen3_hijack import qwen3_attention_forward

    qwen3_attention.Qwen3Attention.forward = qwen3_attention_forward
def replace_glm(model):
    transformers_version = check_version()
    version_list = ['4.57']
    warning_flag = True
    for v in version_list:
        if v in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    enable_snapkv_glm_attention(model)
