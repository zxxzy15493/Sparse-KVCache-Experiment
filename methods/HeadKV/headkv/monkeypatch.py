from importlib.metadata import version
import warnings
import transformers

from headkv.adaptive_llama_hijack import reason_llama_flash_attn2_forward, adaptive_llama_flash_attn2_forward,adaptive_LlamaModel_forward
from headkv.adaptive_llama_hijack import prepare_inputs_for_generation_llama as ada_prepare_inputs_for_generation_llama

from headkv.adaptive_qwen2_hijack import reason_qwen2_flash_attn2_forward, adaptive_qwen2_flash_attn2_forward,adaptive_qwen2Model_forward
from headkv.adaptive_qwen2_hijack import prepare_inputs_for_generation_qwen2 as ada_prepare_inputs_for_generation_qwen2

from headkv.chatglm_hijack import (
    reason_chatglm_flash_attn2_forward,
    adaptive_chatglm_flash_attn2_forward,
    chatglm_transformer_forward_no_cat,
)


def _normalize_method(method: str) -> str:
    if method is None:
        raise ValueError("method must be provided")
    cleaned = "".join(ch for ch in str(method) if ch.isalnum()).lower()
    mapping = {
        "adativekv": "AdativeKV",
        "adaptivekv": "AdativeKV",
        "reasonkv": "ReasonKV",
        "snapkv": "SnapKV",
        "pyramidkv": "PyramidKV",
        "fullkv": "fullkv",
    }
    if cleaned not in mapping:
        raise ValueError(f"Unsupported method: {method}")
    return mapping[cleaned]

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.37', '4.38', '4.39', '4.40', '4.41', '4.42', '4.43', '4.44', '4.45']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")




def replace_llama_adaptive():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaAttention.forward = adaptive_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward


def replace_qwen2_adaptive():
    check_version()
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_qwen2
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = adaptive_qwen2_flash_attn2_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = adaptive_qwen2Model_forward






def replace_llama(method):
    check_version()
    method = _normalize_method(method)

    llama_models = transformers.models.llama.modeling_llama

    if method == "AdativeKV":
        llama_models.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
        llama_models.LlamaModel.forward = adaptive_LlamaModel_forward
        llama_models.LlamaFlashAttention2.forward = adaptive_llama_flash_attn2_forward
        llama_models.LlamaAttention.forward = adaptive_llama_flash_attn2_forward
        llama_models.LlamaSdpaAttention.forward = adaptive_llama_flash_attn2_forward
        print("use AdativeKV monkey patch for LLaMA.")
    elif method == "ReasonKV":
        llama_models.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
        llama_models.LlamaModel.forward = adaptive_LlamaModel_forward
        llama_models.LlamaFlashAttention2.forward = reason_llama_flash_attn2_forward
        llama_models.LlamaAttention.forward = reason_llama_flash_attn2_forward
        llama_models.LlamaSdpaAttention.forward = reason_llama_flash_attn2_forward
        print("use ReasonKV monkey patch for LLaMA.")
    elif method == 'SnapKV':
        llama_models.LlamaForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_llama
        llama_models.LlamaModel.forward = fixed_LlamaModel_forward
        llama_models.LlamaFlashAttention2.forward = fixed_llama_flash_attn2_forward
        llama_models.LlamaAttention.forward = fixed_llama_flash_attn2_forward
        llama_models.LlamaSdpaAttention.forward = fixed_llama_flash_attn2_forward
    elif method == 'PyramidKV':
        llama_models.LlamaForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_llama
        llama_models.LlamaModel.forward = fixed_LlamaModel_forward
        llama_models.LlamaFlashAttention2.forward = pyramidkv_llama_flash_attn2_forward
        llama_models.LlamaAttention.forward = pyramidkv_llama_flash_attn2_forward
        llama_models.LlamaSdpaAttention.forward = pyramidkv_llama_flash_attn2_forward


def replace_qwen2(method):
    check_version()
    method = _normalize_method(method)

    qwen2_models = transformers.models.qwen2.modeling_qwen2

    if method == "AdativeKV":
        qwen2_models.Qwen2ForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_qwen2
        qwen2_models.Qwen2Model.forward = adaptive_qwen2Model_forward
        qwen2_models.Qwen2Attention.forward = adaptive_qwen2_flash_attn2_forward
        if hasattr(qwen2_models, 'Qwen2FlashAttention2'):
            qwen2_models.Qwen2FlashAttention2.forward = adaptive_qwen2_flash_attn2_forward
        if hasattr(qwen2_models, 'Qwen2SdpaAttention'):
            qwen2_models.Qwen2SdpaAttention.forward = adaptive_qwen2_flash_attn2_forward
        print("use AdativeKV monkey patch for Qwen2.")
    elif method == "ReasonKV":
        qwen2_models.Qwen2ForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_qwen2
        qwen2_models.Qwen2Model.forward = adaptive_qwen2Model_forward
        qwen2_models.Qwen2Attention.forward = reason_qwen2_flash_attn2_forward
        if hasattr(qwen2_models, 'Qwen2FlashAttention2'):
            qwen2_models.Qwen2FlashAttention2.forward = reason_qwen2_flash_attn2_forward
        if hasattr(qwen2_models, 'Qwen2SdpaAttention'):
            qwen2_models.Qwen2SdpaAttention.forward = reason_qwen2_flash_attn2_forward
        print("use ReasonKV monkey patch for Qwen2.")
    elif method == 'SnapKV':
        qwen2_models.Qwen2ForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_qwen2
        qwen2_models.Qwen2Model.forward = fixed_qwen2Model_forward
        qwen2_models.Qwen2Attention.forward = fixed_qwen2_flash_attn2_forward
        if hasattr(qwen2_models, 'Qwen2FlashAttention2'):
            qwen2_models.Qwen2FlashAttention2.forward = fixed_qwen2_flash_attn2_forward
        if hasattr(qwen2_models, 'Qwen2SdpaAttention'):
            qwen2_models.Qwen2SdpaAttention.forward = fixed_qwen2_flash_attn2_forward
    elif method == 'PyramidKV':
        qwen2_models.Qwen2ForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_qwen2
        qwen2_models.Qwen2Model.forward = fixed_qwen2Model_forward
        qwen2_models.Qwen2Attention.forward = pyramidkv_qwen2_flash_attn2_forward
        if hasattr(qwen2_models, 'Qwen2FlashAttention2'):
            qwen2_models.Qwen2FlashAttention2.forward = pyramidkv_qwen2_flash_attn2_forward
        if hasattr(qwen2_models, 'Qwen2SdpaAttention'):
            qwen2_models.Qwen2SdpaAttention.forward = pyramidkv_qwen2_flash_attn2_forward


def replace_chatglm(method, model):
    check_version()
    method = _normalize_method(method)

    if model is None or not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
        return

    if method == "AdativeKV":
        target_forward = adaptive_chatglm_flash_attn2_forward
    elif method == "ReasonKV":
        target_forward = reason_chatglm_flash_attn2_forward
    elif method == "SnapKV":
        target_forward = fixed_chatglm_flash_attn2_forward
    elif method == "PyramidKV":
        target_forward = pyramidkv_chatglm_flash_attn2_forward
    else:
        return

    model.transformer.encoder.forward = chatglm_transformer_forward_no_cat.__get__(
        model.transformer.encoder, model.transformer.encoder.__class__
    )

    for layer_idx, layer in enumerate(model.transformer.encoder.layers):
        attn = layer.self_attention
        if not hasattr(attn, "config"):
            attn.config = model.config
        attn.layer_idx = layer_idx
        attn.forward = target_forward.__get__(attn, attn.__class__)