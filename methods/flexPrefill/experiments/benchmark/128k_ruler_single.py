# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
import sys
from tqdm.auto import tqdm
import yaml
import importlib
#from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from flex_prefill import patch_model
from utils import (
    seed_everything,
    get_args,
    str_to_dict,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

accelerator = Accelerator()

SEQ_LENGTHS = ["131072"]

TASKS = [
    "niah_single_1",
    "niah_multiquery",
    "vt",
    "fwe",
]

TASK_TO_MAX_NEW_TOKNES = {
    "niah_single_1": 256,
    "niah_single_2": 256,
    "niah_single_3": 256,
    "niah_multikey_1": 256,
    "niah_multikey_2": 256,
    "niah_multikey_3": 256,
    "niah_multivalue": 256,
    "niah_multiquery": 256,
    "vt": 256,
    "cwe": 256,
    "fwe": 256,
    "qa_1": 256,
    "qa_2": 256,
}


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_dataloader(data_list):
    data_loader = DataLoader(ListDataset(data_list), batch_size=1, shuffle=False)
    return data_loader


def read_manifest(file_path):
    file_path = str(file_path)
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON decode error in {file_path} at line {line_num}: {e}"
                )
    return data


def get_tasks(task_str: str):
    if task_str == "ruler":
        tasks = []
        for t in TASKS:
            for s in SEQ_LENGTHS:
                tasks.append((t, s))
        return tasks
    elif task_str.startswith("ruler"):
        tasks = []
        length = task_str.split(",")[-1]
        for t in TASKS:
            tasks.append((t, length))
        return tasks
    else:
        task, length = task_str.split(",")
        return [(task, length)]


def main():
    args = get_args()
    seed_everything(args.seed)
    model_name = args.model.strip("/").split("/")[-1]

    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    attention_pattern = args.attention
    attention_config = str_to_dict(args.cfg)
    patch_model(model, attention_pattern, attention_config)

    curr_folder = os.path.dirname(os.path.abspath(__file__))
    flex_root = os.path.abspath(os.path.join(curr_folder, "../.."))
    repo_root = os.path.abspath(os.path.join(curr_folder, "../../../.."))
    ruler_data_root = os.path.join(repo_root, "benchmarks", "ruler", "benchmark_root")
    try:
        sys.path.append(os.path.join(curr_folder, "ruler"))
        module = importlib.import_module("data.synthetic.constants")
    except ImportError:
        print("Module data.synthetic.constants not found.")
        return

    tasks_base = module.TASKS
    with open(
        os.path.join(curr_folder, os.path.join(curr_folder, "ruler", "synthetic.yaml")),
        "r",
    ) as f:
        tasks_customized = yaml.safe_load(f)

    # get dataloader
    dataloaders = []
    all_tasks = get_tasks(args.task)
    for task, length in all_tasks:
        if task not in tasks_customized:
            raise ValueError(f"{task} is not found in config_tasks.yaml")
        config = tasks_customized.get(task)
        config.update(tasks_base[config["task"]])
        if "llama" in model_name.lower():
            save_dir = os.path.join(
                flex_root, "outputs", "ruler", "llama-3.1-8b", "synthetic")
            task_file = os.path.join(
                ruler_data_root,
                "llama-3.1-8b",
                "synthetic",
                length,
                "data",
                task,
                "validation.jsonl",
            )
        elif "qwen" in model_name.lower():
            save_dir = os.path.join(
                flex_root, "outputs", "ruler", "qwen-2.5-7b-1m", "synthetic")
            task_file = os.path.join(
                ruler_data_root,
                "qwen-2.5-7b-1m",
                "synthetic",
                length,
                "data",
                task,
                "validation.jsonl",
            )
        else:
            raise ValueError(f"Unknown model type in model_name: {model_name}")


        os.makedirs(os.path.join(save_dir, length), exist_ok=True)
        pred_file = os.path.join(save_dir, length, f"{task}.jsonl")

        # Load data
        data = read_manifest(task_file)
        dataloaders.append(get_dataloader(data))

    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)

    for loader, (task, length) in zip(dataloaders, all_tasks):
        loader = accelerator.prepare_data_loader(loader)
        pred_file = os.path.join(save_dir, length, f"{task}.jsonl")
        def get_output(index, input, outputs, others, truncation, length_val):
            try:
                if args.chat:
                    try:
                        input_ids = tokenizer.apply_chat_template(
                            [{"role": "user", "content": input}],
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(model.device)
                    except Exception:
                        encoded = tokenizer(
                            input,
                            return_tensors="pt",
                        )
                        input_ids = encoded.input_ids.to(model.device)
                        attention_mask = encoded.attention_mask.to(model.device)
                else:
                    encoded = tokenizer(
                        input,
                        return_tensors="pt",
                    )
                    input_ids = encoded.input_ids.to(model.device)
                    attention_mask = encoded.attention_mask.to(model.device)

                do_sample = False if args.top_p <= 0 else True
                generation_config = dict(
                    do_sample=do_sample,
                    max_new_tokens=TASK_TO_MAX_NEW_TOKNES[task],
                    pad_token_id=tokenizer.eos_token_id,
                )
                if do_sample:
                    generation_config["top_p"] = args.top_p
                    generation_config["temperature"] = args.temperature

                if "attention_mask" in locals():
                    output = model.generate(
                        input_ids, attention_mask=attention_mask, **generation_config
                    )
                else:
                    output = model.generate(input_ids, **generation_config)

                generated_text = tokenizer.decode(
                    output[0][input_ids.shape[1] :], skip_special_tokens=True
                )
                # remove the input form the generated text   
                if generated_text.startswith(input):
                    generated_text = generated_text[len(input) :]
                # remove the </s> from llama-3-8b-262k
                if generated_text.find("</s>") > 0:
                    generated_text = generated_text[: generated_text.find("</s>")]
                pred = {"text": [generated_text]}

                if len(pred["text"]) > 0:
                    return {
                        "index": int(index),
                        "pred": pred["text"][0],
                        "input": input,
                        "outputs": outputs,
                        "others": others,
                        "truncation": truncation,
                        "length": length_val,
                    }
                else:
                    return None
            except Exception as e:
                print(
                    f"[WARN] get_output failed at task={task}, length={length}, index={index}, err={e}"
                )
                return None

        if len(loader) == 0:
            print(f"[INFO] No samples to run for task={task}, length={length}")
            continue
        pbar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)

        with open(pred_file, "wt", encoding="utf-8", buffering=1) as fout:
            for idx, data_point in enumerate(loader):
                if data_point is None:
                    print(
                        f"[WARN] data_point is None at task={task}, length={length}, idx={idx}"
                    )
                    continue

                if not isinstance(data_point, dict):
                    print(
                        f"[WARN] Unexpected data_point type at task={task}, length={length}, idx={idx}, type={type(data_point)}"
                    )
                    print("data_point =", data_point)
                    continue

                if "index" not in data_point:
                    print(
                        f"[WARN] 'index' not in data_point at task={task}, length={length}, idx={idx}"
                    )
                    print("data_point =", data_point)
                    continue

                if data_point["index"] is None or len(data_point["index"]) == 0:
                    print(
                        f"[WARN] data_point['index'] is None/empty at task={task}, length={length}, idx={idx}"
                    )
                    print("data_point =", data_point)
                    continue

                invalid_sample = False
                for key in ["input", "outputs"]:
                    if key not in data_point or data_point[key] is None or len(data_point[key]) == 0:
                        print(
                            f"[WARN] key '{key}' missing/empty in data_point at task={task}, length={length}, idx={idx}"
                        )
                        print("data_point =", data_point)
                        invalid_sample = True
                        break
                if invalid_sample:
                    continue

                outputs = [
                    item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else item
                    for item in data_point["outputs"]
                ]
                output = get_output(
                    data_point["index"][0],
                    data_point["input"][0],
                    outputs,
                    data_point.get("others", [{}])[0],
                    data_point.get("truncation", [-1])[0],
                    int(data_point.get("length", [-1])[0]),
                )

                if output is not None:
                    if accelerator.is_main_process:
                        fout.write(json.dumps(output) + "\n")

                pbar.set_description(desc=f"task {task}, len {length}")
                pbar.update(1)

        accelerator.wait_for_everyone()

    all_length = set([length for _, length in all_tasks])

    for length in all_length:
        if accelerator.is_main_process:
            pred_dir = os.path.join(save_dir, length)
            evaluate_py = os.path.join(
                flex_root, "experiments", "benchmark", "ruler", "eval", "evaluate.py"
            )
            cmd = (
                f"{sys.executable} {evaluate_py} "
                f"--data_dir {pred_dir} --benchmark synthetic"
            )
            os.system(cmd)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
