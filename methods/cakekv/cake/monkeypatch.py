import transformers
from cakekv.cake.model.modify_llama import llama_model_forward_cake, llama_attn_forward_cake
from cakekv.cake.model.modify_mistral import mistral_model_forward_cake, mistral_attn_forward_cake
from cakekv.cake.model.modify_qwen2 import qwen2_model_forward_cake, qwen2_attn_forward_cake
from cakekv.cake.model.modify_chatglm import chatglm_model_forward_cake, chatglm_attn_forward_cake
    

def replace_flashllama_attn_with_cakeattn():
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_cake
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward_cake

def replace_flashmistral_attn_with_cakeattn():
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_cake
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_attn_forward_cake

def replace_flashqwen2_attn_with_cakeattn():
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward_cake
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_attn_forward_cake


def replace_chatglm_attn_with_cakeattn(model):
    type(model.transformer).forward = chatglm_model_forward_cake
    type(model.transformer.encoder.layers[0].self_attention).forward = chatglm_attn_forward_cake

