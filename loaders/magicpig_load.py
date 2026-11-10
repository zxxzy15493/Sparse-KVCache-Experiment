from benchmarks.common import method_defaults, parse_set

import torch    
import sys
from pathlib import Path
from types import MethodType
from benchmarks.common import MODEL_CONTEXT_LENGTHS, model_family
from loaders.chunked_post_ffn import enable_chunked_post_ffn


def _has_avx512f() -> bool:
    """MagicPig's bundled CPU extensions are compiled with ``-mavx512f``."""
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").lower().splitlines()
    except OSError:
        return False
    flags = [line.split(":", 1)[1].split() for line in lines
             if line.partition(":")[0].strip() == "flags"]
    return bool(flags) and all("avx512f" in cpu_flags for cpu_flags in flags)


def _magicpig_model_key(model):
    family = model_family(model)
    return "glm" if family == "glm4" else family


def _magicpig_kl_config(args, dataset):
    defaults = method_defaults("magicpig")
    budget_key = f"budget-{args.budget}"
    if budget_key not in defaults:
        available = ", ".join(sorted(defaults))
        raise ValueError(f"Missing MagicPig config for {budget_key}. Available budgets: {available}")

    model_key = _magicpig_model_key(args.model)
    budget_config = defaults[budget_key]
    model_config = budget_config.get(model_key)
    if model_config is None and model_key == "qwen":
        model_config = budget_config.get("llama")
    if model_config is None:
        available = ", ".join(sorted(budget_config))
        raise ValueError(f"Missing MagicPig config for model '{args.model}' ({model_key}). Available models: {available}")

    dataset_config = model_config.get(dataset)
    if dataset_config is None and dataset.startswith("ruler-") and dataset.split("-", 1)[1].isdigit():
        ruler_lengths = {
            int(key.split("-", 1)[1]): key
            for key in model_config
            if key.startswith("ruler-") and key.split("-", 1)[1].isdigit()
        }
        requested = int(dataset.split("-", 1)[1])
        candidates = [length for length in ruler_lengths if length <= requested]
        if candidates:
            dataset_config = model_config[ruler_lengths[max(candidates)]]
    if dataset_config is None:
        available = ", ".join(sorted(model_config))
        raise ValueError(f"Missing MagicPig config for dataset '{dataset}'. Available datasets: {available}")

    return dataset_config["K"], dataset_config["L"]


def load_model_and_tokenizer(args):
    if args.dataset is None:
        return None, None
    use_cpu_backend = _has_avx512f()
    batch_size = int(getattr(args, "batch_size", 1))
    if batch_size < 1:
        raise ValueError("MagicPig batch_size must be positive.")
    generation_buffer = max(256, int(getattr(args, "fixed_output_length", 0) or 0))
    
    magic_root = Path(__file__).resolve().parents[1] / "methods" / "magicpig" 
    if str(magic_root) not in sys.path:
        sys.path.insert(0, str(magic_root))

    from transformers import AutoTokenizer
    if args.dataset == "longbenchv2":
        K=9
        L=115
        max_len = 192*1024
    elif args.dataset == "gsm8k":
        K=7
        L=200
        max_len = 6000
    elif args.dataset == "ruler":
        K, L = _magicpig_kl_config(args, f"ruler-{args.max_seq_length}")
        max_len = args.max_seq_length + 4096
    else:
        K, L = _magicpig_kl_config(args, args.dataset)
        max_len = 65536

    
    # print K_L 参数
    backend = "models_single (CPU LSH + GPU)" if use_cpu_backend else "models_gpu (HF GPU Mask)"
    print(f"run {args.dataset}, use K = {K}, L = {L} on {args.model_path}, backend = {backend}")
    if not use_cpu_backend:
        if 'qwen' not in args.model_path.lower():
            raise ValueError(
                "MagicPig models_gpu currently supports dense Qwen2/Qwen3, including "
                "Qwen/Qwen2.5-7B-Instruct-1M. This CPU lacks AVX-512F; "
                f"the models_single backend cannot load {args.model_path!r}."
            )
        # Never import models_single or its AVX-512 extensions on this path.
        from models_gpu.qwen import Qwen2Model
        llm = Qwen2Model(
            model_name=args.model_path, K=K, L=L, batch_size=batch_size,
            generation_buffer=generation_buffer, max_length=max_len,
            device=getattr(args, "device", "cuda:0"), dtype=torch.bfloat16,
        )
        enable_chunked_post_ffn(llm.model)
    elif 'llama' in args.model_path.lower():
        from models_single.llama import LlamaModel
        llm = LlamaModel(
                model_name=args.model_path, 
                K=K,
                L=L,
                batch_size=batch_size,
                generation_buffer=generation_buffer,
                max_length=max_len, 
                device="cuda:0", 
                dtype=torch.bfloat16,
                )
    elif 'deepseek' in args.model_path.lower():
        from models_single.deepseek import DeepSeekModel
        llm = DeepSeekModel(
                model_name=args.model_path, 
                K=K,
                L=L,
                batch_size=batch_size,
                max_length=max_len, 
                device="cuda:0", 
                dtype=torch.bfloat16,
                generation_buffer=max(3500, generation_buffer)
                )
    elif 'qwen' in args.model_path.lower():
        from models_single.qwen import Qwen2Model
        llm = Qwen2Model(
                model_name=args.model_path, 
                K=K,
                L=L,
                batch_size=batch_size,
                generation_buffer=generation_buffer,
                max_length=max_len,  
                device="cuda:0", 
                dtype=torch.bfloat16,
              )
    elif 'glm' in args.model_path.lower():
        from models_single.glm import GLMModel
        llm = GLMModel(
                model_name=args.model_path,
                K=K,
                L=L,
                batch_size=batch_size,
                generation_buffer=generation_buffer,
                max_length=max_len, 
                device="cuda:0", 
                dtype=torch.bfloat16,
              )
    else:
        raise ValueError(f"Unsupported model: {args.model_path}")


    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    llm._magicpig_backend = "cpu" if use_cpu_backend else "gpu"

    llm._magicpig_original_generate = llm.generate
    llm.generate = MethodType(generate, llm)
    
    return llm, tokenizer


def generate(self, *args, **kwargs):
    input_ids = kwargs.pop("input_ids", None)
    if input_ids is None:
        input_ids = kwargs.pop("inputs_ids", None)
    if input_ids is None and args:
        input_ids = args[0]
        args = args[1:]
    if input_ids is None:
        raise TypeError("MagicPig generate requires `input_ids`.")

    max_new_tokens = kwargs.pop("max_new_tokens", None)
    if max_new_tokens is None:
        max_new_tokens = kwargs.pop("max_tokens", None)
    if max_new_tokens is None:
        max_new_tokens = kwargs.pop("max_new_length", None)
    if max_new_tokens is None:
        max_new_tokens = 128

    attention_mask = kwargs.pop("attention_mask", None)
    if attention_mask is None:
        attention_mask = kwargs.pop("attention_masks", None)
    # Accept the existing HuggingFace-style greedy generation kwargs.
    kwargs.pop("num_beams", None)
    kwargs.pop("do_sample", None)
    kwargs.pop("temperature", None)
    kwargs.pop("min_length", None)
    eos_token_id = kwargs.pop("eos_token_id", None)
    pad_token_id = kwargs.pop("pad_token_id", None)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

    generated = self._magicpig_original_generate(
        input_ids=input_ids.to(self.device),
        max_tokens=int(max_new_tokens),
        attention_mask=attention_mask,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    if isinstance(generated, torch.Tensor):
        output = generated.to(device=input_ids.device, dtype=input_ids.dtype)
    else:
        output = torch.tensor(generated, dtype=input_ids.dtype, device=input_ids.device)

    if output.dim() == 1:
        output = output.unsqueeze(0)
    return output
