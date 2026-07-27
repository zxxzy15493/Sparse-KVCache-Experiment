from importlib.metadata import version
import transformers


def replace_llama(method):
    from .llama_model import llama_flash_attn2_forward_PyramidKV
    from .llama_model import llama_attn_forward_PyramidKV
    from .llama_model import prepare_inputs_for_generation_llama, prepare_inputs_for_generation_llama_new

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV

    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new



def replace_qwen2(method):
    from .qwen2_model import (
        qwen2_flash_attn2_forward_PyramidKV,
        fixed_qwen2Model_forward,
        prepare_inputs_for_generation_qwen2,
    )

    qwen2_models = transformers.models.qwen2.modeling_qwen2
    
    if method == "pyramidkv":
        print("Using PyramidKV!")
        if hasattr(qwen2_models, 'Qwen2FlashAttention2'):
            qwen2_models.Qwen2FlashAttention2.forward = qwen2_flash_attn2_forward_PyramidKV
        if hasattr(qwen2_models, 'Qwen2SdpaAttention'):
            qwen2_models.Qwen2SdpaAttention.forward = qwen2_flash_attn2_forward_PyramidKV
        if hasattr(qwen2_models, 'Qwen2Attention'):
            qwen2_models.Qwen2Attention.forward = qwen2_flash_attn2_forward_PyramidKV
        qwen2_models.Qwen2Model.forward = fixed_qwen2Model_forward

    if method not in ["fullkv"]:
        qwen2_models.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2


def replace_chatglm(method, model):
    from .glm_model import enable_pyramidkv_glm_attention, chatglm_transformer_forward_no_cat

    if model is None or not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
        return

    if method == "pyramidkv":
        print("Using PyramidKV on ChatGLM!")
    else:
        return

    model.transformer.encoder.forward = chatglm_transformer_forward_no_cat.__get__(
        model.transformer.encoder, model.transformer.encoder.__class__
    )

    enable_pyramidkv_glm_attention(model)
