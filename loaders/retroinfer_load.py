import torch    
import sys
from pathlib import Path
from types import MethodType
from benchmarks.common import MODEL_CONTEXT_LENGTHS


def generate(self, *args, **kwargs):
    """Adapt HuggingFace-style generate calls to RetroInfer's generate API."""
    retroinfer_root = Path(__file__).resolve().parents[1] / "methods" / "retroinfer" 
    if str(retroinfer_root) not in sys.path:
        sys.path.insert(0, str(retroinfer_root))

    from config import generate_config

    input_ids = kwargs.pop("input_ids", None)
    if input_ids is None:
        input_ids = kwargs.pop("inputs_ids", None)
    if input_ids is None and args:
        input_ids = args[0]
        args = args[1:]
    if input_ids is None:
        raise TypeError("RetroInfer generate requires `input_ids`.")

    attention_mask = kwargs.pop("attention_mask", None)
    if attention_mask is None:
        attention_mask = kwargs.pop("attention_masks", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    max_new_tokens = kwargs.pop("max_new_tokens", None)
    if max_new_tokens is None:
        max_new_tokens = kwargs.pop("max_new_length", None)
    if max_new_tokens is None:
        max_new_tokens = getattr(self, "max_new_length", None)
    if max_new_tokens is None:
        raise TypeError("RetroInfer generate requires `max_new_tokens` or `max_new_length`.")

    loader_args = getattr(self, "_retroinfer_loader_args", None)
    attention_type = kwargs.pop("attention_type", getattr(loader_args, "attn_type", "RetroInfer"))
    prefill_method = kwargs.pop("prefill_method", getattr(loader_args, "prefill_method", "Full_Flash_Attn"))
    fixed_output_length = kwargs.pop("fixed_output_length", 0)
    eos_token_id = kwargs.pop("eos_token_id", None)

    eos_tokens = getattr(self, "eos_tokens", [])
    eos_token_list = eos_tokens if isinstance(eos_tokens, (list, tuple, set)) else [eos_tokens]

    if self.tokenizer.eos_token is not None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
    elif self.tokenizer.pad_token_id is None and len(eos_token_list) > 0:
        self.tokenizer.pad_token_id = eos_token_list[0]
    self.tokenizer.padding_side = "left"

    attn_config = generate_config(
        getattr(loader_args, "model_path", self.model_name),
        input_ids.shape[1],
        attention_type,
        budget_ratio=getattr(loader_args, "budget_ratio", 0.018),
        budget=getattr(loader_args, "budget", 1024),
        estimate_ratio=getattr(loader_args, "estimate_ratio", 0.25),
        ratio_or_fixed=getattr(loader_args, "ratio_or_fixed", 1),
    )

    original_eos_tokens = getattr(self, "eos_tokens", [])
    original_eos_token_list = (
        original_eos_tokens
        if isinstance(original_eos_tokens, (list, tuple, set))
        else [original_eos_tokens]
    )
    if eos_token_id is not None:
        if isinstance(eos_token_id, torch.Tensor):
            eos_token_id = eos_token_id.detach().cpu().tolist()
        if not isinstance(eos_token_id, (list, tuple, set)):
            eos_token_id = [eos_token_id]
        self.eos_tokens = list(dict.fromkeys(list(original_eos_token_list) + list(eos_token_id)))

    original_generate = getattr(self, "_retroinfer_original_generate")
    try:
        output_ids = original_generate(
            attention_type=attention_type,
            inputs_ids=input_ids.to(self.layers[0].device),
            attention_masks=attention_mask.to(self.layers[0].device),
            max_new_length=max_new_tokens,
            attn_config=attn_config,
            prefill_method=prefill_method,
            fixed_output_length=fixed_output_length,
        )
    finally:
        self.eos_tokens = original_eos_tokens

    output_ids = torch.tensor(output_ids, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids, output_ids], dim=-1)


def load_model_and_tokenizer(args):
    retroinfer_root = Path(__file__).resolve().parents[1] / "methods" / "retroinfer" 
    if str(retroinfer_root) not in sys.path:
        sys.path.insert(0, str(retroinfer_root))

    from model_hub import LlamaModel, QwenModel, GlmModel
    
    max_len = MODEL_CONTEXT_LENGTHS[args.model]

    if 'Llama' in args.model_path:
        llm = LlamaModel(
            args.model_path,
            max_length=max_len,
            dtype=torch.bfloat16,
            device_map="auto",
            )
    elif 'Qwen' in args.model_path:
        llm = QwenModel(
            args.model_path,
            max_length=max_len,
            dtype=torch.bfloat16,
            device_map="auto",
            )
    elif 'GLM' in args.model_path or 'glm' in args.model_path:
        llm = GlmModel(
            args.model_path,
            max_length=max_len,
            dtype=torch.bfloat16,
            device_map="auto",
            )
    else:
        raise ValueError(f"Unsupported model: {args.model_path}")

    if llm.tokenizer.eos_token is not None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
        llm.tokenizer.pad_token_id = llm.eos_tokens[0]
    llm.tokenizer.padding_side = "left"

    llm._retroinfer_loader_args = args
    llm._retroinfer_original_generate = llm.generate
    llm.generate = MethodType(generate, llm)

    return llm, llm.tokenizer
