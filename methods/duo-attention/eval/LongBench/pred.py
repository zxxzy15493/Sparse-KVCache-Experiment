import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

from duo_attn.patch import enable_duo_attention_eval

from duo_attn.utils import (
    to_device,
    load_attn_pattern,
    sparsify_attention_heads,
)
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")

    parser.add_argument("--task", type=str, help="task name", required=True)

    parser.add_argument(
        "--method",
        type=str,
        default="full",
    )

    # duo attention
    parser.add_argument(
        "--attn_load_dir", type=str, default=None, help="attention pattern directory"
    )
    parser.add_argument("--sink_size", type=int, default=64)
    parser.add_argument("--recent_size", type=int, default=256)

    parser.add_argument("--sparsity", type=float, default=0.5) 

    parser.add_argument("--decoding_simulation_length", type=int, default=50)
    parser.add_argument("--max_num_examples", type=int, default=None)

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                             add_generation_prompt=True, tokenize=False)
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
    return response


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
    decoding_simulation_length,
    out_path=None,
):
    preds = []
    pbar = tqdm(data)

    avg_length = np.mean([ex.get("length", 0) for ex in data]) if len(data) > 0 else 0
    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        seq_len = input.input_ids.shape[-1]
        pbar.set_description(
            f"Generating for {idx}, len = {seq_len}"
        )

        

        simulation_start_idx = max(
            0, seq_len - decoding_simulation_length
        )
        with torch.no_grad():
            model_extra_kwargs = {}
            is_glm = "chatglm" in model_name.lower() or "glm" in model_name.lower()
            if is_glm:
                model_extra_kwargs["return_last_logit"] = True
            if "qwen2" in model_name.lower():
                # Only need the last-token logits during decoding; avoid full-logits OOM.
                model_extra_kwargs["num_logits_to_keep"] = 1

            past_key_values = None
            last_logits = None
            current_pos = 0  # track position for ChatGLM RoPE

            if simulation_start_idx > 0:
                output = model(
                    input_ids=input.input_ids[:, :simulation_start_idx],
                    past_key_values=past_key_values,
                    use_cache=True,
                    **model_extra_kwargs,
                )
                past_key_values = output.past_key_values
                last_logits = output.logits[:, -1, :]
                if is_glm:
                    current_pos = simulation_start_idx

            if decoding_simulation_length > 0:
                for idx, input_id in enumerate(
                    input.input_ids[0, simulation_start_idx:]
                ):
                    kwargs = model_extra_kwargs.copy()
                    if is_glm:
                        kwargs["position_ids"] = torch.tensor([[current_pos]], device="cuda")
                        current_pos += 1
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                        **kwargs,
                    )
                    past_key_values = output.past_key_values
                    last_logits = output.logits[:, -1, :]

            if last_logits is None:
                output = model(
                    input_ids=input.input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    **model_extra_kwargs,
                )
                past_key_values = output.past_key_values
                last_logits = output.logits[:, -1, :]
                if is_glm:
                    current_pos = input.input_ids.shape[-1]

            pred_token_idx = last_logits.argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                kwargs = model_extra_kwargs.copy()
                if is_glm:
                    kwargs["position_ids"] = torch.tensor([[current_pos]], device="cuda")
                    current_pos += 1
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                    **kwargs,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in eos_token_ids:
                    break

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        print(f"Prediction: {pred}")
        pred_dict = {
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        }
        preds.append(pred_dict)


        if out_path is not None:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(pred_dict, f, ensure_ascii=False)
                f.write("\n")

    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    
    if 'glm' in model_name.lower() or 'chatglm' in model_name.lower():
        import sys
        import importlib.util
        
        modeling_path = os.path.join(path, "modeling_chatglm.py")
        if os.path.exists(modeling_path):
            spec = importlib.util.spec_from_file_location("modeling_chatglm", modeling_path)
            modeling_chatglm = importlib.util.module_from_spec(spec)
            sys.modules["modeling_chatglm"] = modeling_chatglm
            spec.loader.exec_module(modeling_chatglm)
            
            config_path = os.path.join(path, "configuration_chatglm.py")
            if os.path.exists(config_path):
                spec = importlib.util.spec_from_file_location("configuration_chatglm", config_path)
                configuration_chatglm = importlib.util.module_from_spec(spec)
                sys.modules["configuration_chatglm"] = configuration_chatglm
                spec.loader.exec_module(configuration_chatglm)
            
            config = configuration_chatglm.ChatGLMConfig.from_pretrained(path)
            model = modeling_chatglm.ChatGLMForConditionalGeneration.from_pretrained(
                path,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                local_files_only=True,
            )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                local_files_only=True,
            )
        except ValueError as e:
            if "config_class" in str(e):
                print(f"Caught config_class error: {e}")
                print("Trying to load without attn_implementation...")
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            else:
                raise

    generation_config = GenerationConfig.from_pretrained(path, local_files_only=True)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()
    
    model.gradient_checkpointing_enable()

    if args.method == "duo_attn":
        assert args.attn_load_dir is not None, "attn_load_dir must be provided"
        full_attention_heads, _, _ = load_attn_pattern(args.attn_load_dir)
        sink_size = args.sink_size
        recent_size = args.recent_size
        print(f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}")

        mask, sparsity = sparsify_attention_heads(
            full_attention_heads.copy(), None, sparsity=args.sparsity
        )
        print(f"True sparsity: {sparsity}")

        enable_duo_attention_eval(
            model,
            torch.tensor(mask, dtype=torch.float32),
            sink_size,
            recent_size,
        )
    else:
        enable_tuple_kv_cache(model)

    return model, tokenizer, eos_token_ids


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model
    # define your model
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model2path[model_name], model_name
    )
    if 'glm' not in model_name.lower() and 'chatglm' not in model_name.lower():
        model = to_device(model, device_list, enable_tp=True)
    max_length = model2maxlen[model_name]
    if args.e:
        # datasets = [
        #     "qasper",
        #     "multifieldqa_en",
        #     "hotpotqa",
        #     "2wikimqa",
        #     "gov_report",
        #     "multi_news",
        #     "trec",
        #     "triviaqa",
        #     "samsum",
        #     "passage_count",
        #     "passage_retrieval_en",
        #     "lcc",
        #     "repobench-p",
        # ]
                datasets = [
            "passage_count"
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        data = load_dataset("THUDM/LongBench", dataset, split="test")
        data = [item for item in data]  # convert Dataset to list for indexing/slicing

        if args.max_num_examples is not None:
            data = data[: args.max_num_examples]
        
        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        if args.method == "duo_attn":
            out_path = f"pred/{model_name}/{dataset}-duo_attn-pattern-{args.attn_load_dir.split('/')[-1]}-sp{args.sparsity}.jsonl"
        else:
            out_path = f"pred/{model_name}/{dataset}-full.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            if len(data) == len(lines):
                print(f"[SKIP] {dataset} 已完成，跳过")
                continue
            elif len(lines) > 0:
                print(f"[RESUME] {dataset} 从第 {len(lines)}/{len(data)} 个样本继续")
                data = data[len(lines):]
        else:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        preds = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
            args.decoding_simulation_length,
            out_path=out_path,
        )


        if not os.path.exists(out_path):
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write("\n")