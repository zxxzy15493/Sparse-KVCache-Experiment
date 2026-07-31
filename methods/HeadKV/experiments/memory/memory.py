import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
	parser = argparse.ArgumentParser(description="HeadKV memory benchmark")
	parser.add_argument("--model", type=str, default=None, help="model path")
	parser.add_argument(
		"--models",
		nargs="+",
		default=None,
	)
	parser.add_argument("--input_file", type=str, required=True)
	parser.add_argument("--save_dir", type=str, default="./results")
	parser.add_argument("--warmup", type=int, default=3)
	parser.add_argument("--input_max_token", type=int, default=1024)
	parser.add_argument(
		"--input_max_tokens",
		nargs="+",
		type=int,
		default=None,
	)
	parser.add_argument("--max_new_tokens", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--use_fast_tokenizer", action="store_true")
	parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])

	parser.add_argument("--method", type=str, required=True, help="fullkv/SnapKV/PyramidKV/AdativeKV/ReasonKV")
	parser.add_argument("--max_capacity_prompts", type=int, default=512)
	parser.add_argument("--head_choice", type=str, default="random", choices=["random", "copy", "reason"])
	parser.add_argument("--beta", type=float, default=1.5)
	parser.add_argument("--temp", type=float, default=1.0)

	args = parser.parse_args()
	if not args.model and not args.models:
		parser.error("must specify either --model or --models")
	return args


def normalize_model_list(args):
	if args.models:
		return args.models
	return [args.model]


def normalize_input_lengths(args):
	if args.input_max_tokens:
		return args.input_max_tokens
	return [args.input_max_token]


def make_safe_name(name: str) -> str:
	base = os.path.basename(str(name).rstrip("/"))
	return base.replace(" ", "_")


def seed_everything(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def load_dataset(input_file: str) -> str:
	with open(input_file, "r", encoding="utf-8") as f:
		return f.read()


def get_transformer_layers(model):
	if hasattr(model, "model") and hasattr(model.model, "layers"):
		return model.model.layers
	if (
		hasattr(model, "transformer")
		and hasattr(model.transformer, "encoder")
		and hasattr(model.transformer.encoder, "layers")
	):
		return model.transformer.encoder.layers
	raise ValueError("Unsupported model architecture for layer traversal")


def get_self_attention_module(layer):
	if hasattr(layer, "self_attn"):
		return layer.self_attn
	if hasattr(layer, "self_attention"):
		return layer.self_attention
	raise ValueError("Unsupported layer attention module")


def get_config_holder(model):
	if hasattr(model, "model") and hasattr(model.model, "config"):
		return model.model.config
	if hasattr(model, "transformer") and hasattr(model.transformer, "config"):
		return model.transformer.config
	return model.config


def ensure_pad_token(tokenizer, model):
	tokenizer.padding_side = "left"
	tokenizer.truncation_side = "left"
	if tokenizer.pad_token is None:
		if tokenizer.eos_token is not None:
			tokenizer.pad_token = tokenizer.eos_token
			tokenizer.pad_token_id = tokenizer.eos_token_id
		else:
			tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
			model.resize_token_embeddings(len(tokenizer))


def apply_headkv_patch(args, model):
	method_lower = args.method.lower()
	if method_lower == "fullkv":
		return

	from headkv.monkeypatch import replace_chatglm, replace_llama, replace_qwen2

	replace_llama(args.method)
	replace_qwen2(args.method)

	if "glm" in args.model.lower() or "chatglm" in args.model.lower():
		replace_chatglm(args.method, model)


def apply_headkv_runtime_config(model, args):
	layers_list = get_transformer_layers(model)
	config_holder = get_config_holder(model)

	config_holder.window_size = 8
	config_holder.base_capacity = args.max_capacity_prompts
	config_holder.head_choice = args.head_choice
	config_holder.beta = args.beta
	config_holder.temp = args.temp

	config_holder.kernel_size = 7
	config_holder.skip = 0
	config_holder.normalize = True
	config_holder.pooling = "maxpool"
	config_holder.floor = 0.2

	for layer in layers_list:
		attn_module = get_self_attention_module(layer)
		if not hasattr(attn_module, "config"):
			attn_module.config = model.config
		attn_module.config.window_size = config_holder.window_size
		attn_module.config.base_capacity = config_holder.base_capacity
		attn_module.config.head_choice = config_holder.head_choice
		attn_module.config.beta = config_holder.beta
		attn_module.config.temp = config_holder.temp
		attn_module.config.kernel_size = config_holder.kernel_size
		attn_module.config.skip = config_holder.skip
		attn_module.config.normalize = config_holder.normalize
		attn_module.config.pooling = config_holder.pooling
		attn_module.config.floor = config_holder.floor


def load_model_and_tokenizer(args):
	tokenizer = AutoTokenizer.from_pretrained(
		args.model,
		use_fast=args.use_fast_tokenizer,
		trust_remote_code=True,
		padding_side="left",
	)

	model = None
	model_name_lower = args.model.lower()

	apply_headkv_patch(args, model=None)

	if "glm" in model_name_lower or "chatglm" in model_name_lower:
		config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
		config.use_cache = True
		config._attn_implementation = args.attn_implementation
		chatglm_cls = get_class_from_dynamic_module(
			"modeling_chatglm.ChatGLMForConditionalGeneration",
			args.model,
			local_files_only=True,
		)
		model = chatglm_cls.from_pretrained(
			args.model,
			config=config,
			torch_dtype=torch.bfloat16,
			low_cpu_mem_usage=True,
			device_map="auto",
			trust_remote_code=True,
			local_files_only=True,
		)
		apply_headkv_patch(args, model)
	else:
		model = AutoModelForCausalLM.from_pretrained(
			args.model,
			torch_dtype=torch.bfloat16,
			low_cpu_mem_usage=True,
			device_map="auto",
			use_cache=True,
			attn_implementation=args.attn_implementation,
			trust_remote_code=True,
		)

	model.eval()
	ensure_pad_token(tokenizer, model)

	model.config.use_cache = True
	if hasattr(model, "generation_config"):
		model.generation_config.use_cache = True

	apply_headkv_runtime_config(model, args)
	return model, tokenizer


def measure_memory(model, tokenizer, text, args):
	os.makedirs(args.save_dir, exist_ok=True)

	inputs = tokenizer(
		[text],
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=args.input_max_token,
		add_special_tokens=True,
	)

	device = next(model.parameters()).device
	input_ids = inputs.input_ids.to(device)
	attention_mask = inputs.attention_mask.to(device)

	print("=" * 80)
	print(f"Input tokens: {input_ids.shape[1]}")
	print("=" * 80)

	warmup_count = 2
	measure_count = 2

	eos_token_ids = tokenizer.eos_token_id
	if eos_token_ids is None:
		eos_token_ids = getattr(model.config, "eos_token_id", None)
	if isinstance(eos_token_ids, int):
		eos_token_ids = [eos_token_ids]
	if eos_token_ids is None:
		eos_token_ids = []

	model_extra_kwargs = {}
	if "chatglm" in args.model.lower() or "glm" in args.model.lower():
		model_extra_kwargs["return_last_logit"] = True
	model_name_lower = args.model.lower()
	if "llama" in model_name_lower or "qwen" in model_name_lower or "mistral" in model_name_lower:
		model_extra_kwargs["num_logits_to_keep"] = 1

	# Warmup runs
	for _ in range(warmup_count):
		torch.cuda.empty_cache()
		torch.cuda.synchronize()

		with torch.inference_mode():
			prefill_out = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				past_key_values=None,
				use_cache=True,
				**model_extra_kwargs,
			)
			past_key_values = prefill_out.past_key_values
			pred_token_idx = prefill_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
			torch.cuda.synchronize()

			for _ in range(max(args.max_new_tokens - 1, 0)):
				outputs = model(
					input_ids=pred_token_idx,
					past_key_values=past_key_values,
					use_cache=True,
					**model_extra_kwargs,
				)
				torch.cuda.synchronize()
				past_key_values = outputs.past_key_values
				pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
				if pred_token_idx.item() in eos_token_ids:
					break

	torch.cuda.memory.reset_peak_memory_stats()
	torch.cuda.memory._record_memory_history()

	# Measured runs
	for i in range(measure_count):
		torch.cuda.empty_cache()
		torch.cuda.synchronize()

		with torch.inference_mode():
			prefill_out = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				past_key_values=None,
				use_cache=True,
				**model_extra_kwargs,
			)
			past_key_values = prefill_out.past_key_values
			pred_token_idx = prefill_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
			torch.cuda.synchronize()

			generated_ids = [pred_token_idx.item()]

			for _ in range(max(args.max_new_tokens - 1, 0)):
				outputs = model(
					input_ids=pred_token_idx,
					past_key_values=past_key_values,
					use_cache=True,
					**model_extra_kwargs,
				)
				torch.cuda.synchronize()

				past_key_values = outputs.past_key_values
				pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
				generated_ids.append(pred_token_idx.item())
				if pred_token_idx.item() in eos_token_ids:
					break

		print(f"[Measured Round {i}] Generated {len(generated_ids)} tokens")

	peak_alloc = torch.cuda.memory.max_memory_allocated() / 1024**2

	summary = {
		"peak_memory_mib": peak_alloc,
		"effective_rounds": measure_count,
	}

	return summary


def main():
	args = parse_args()
	seed_everything(args.seed)

	text = load_dataset(args.input_file)
	for model_name in normalize_model_list(args):
		model_args = argparse.Namespace(**vars(args))
		model_args.model = model_name
		model_slug = make_safe_name(model_name)
		model, tokenizer = load_model_and_tokenizer(model_args)
		for input_max_token in normalize_input_lengths(args):
			run_args = argparse.Namespace(**vars(model_args))
			run_args.input_max_token = input_max_token
			run_dir = os.path.join(args.save_dir, model_slug, f"input_{input_max_token}")
			run_args.save_dir = run_dir
			summary = measure_memory(model, tokenizer, text, run_args)

			out_file = os.path.join(
				run_dir,
				f"memory_{model_args.method}_{model_args.max_capacity_prompts}_{input_max_token}_{model_args.max_new_tokens}.json",
			)

			meta = {
				"model": model_name,
				"method": model_args.method,
				"input_file": model_args.input_file,
				"input_max_token": input_max_token,
				"max_new_tokens": model_args.max_new_tokens,
				"warmup": model_args.warmup,
				"max_capacity_prompts": model_args.max_capacity_prompts,
				"head_choice": model_args.head_choice,
				"beta": model_args.beta,
				"temp": model_args.temp,
				"attn_implementation": model_args.attn_implementation,
			}

			os.makedirs(run_dir, exist_ok=True)
			with open(out_file, "w", encoding="utf-8") as f:
				json.dump({"meta": meta, "summary": summary}, f, ensure_ascii=False, indent=2)

			print("=" * 80)
			print(f"Saved: {out_file}")
			print(f"Peak Memory: {summary['peak_memory_mib']:.2f} MiB")
			print("=" * 80)


if __name__ == "__main__":
	main()