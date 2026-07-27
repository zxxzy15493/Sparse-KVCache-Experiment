
import os, csv, json, argparse, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import h2o_kv.h2o_time as h2o_core

def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--heavy_hitter_size', type=int, default=992)
    p.add_argument('--recent_size', type=int, default=32)
    p.add_argument('--e', action='store_true')
    p.add_argument('--dataset', type=str, default="myinput.txt")
    p.add_argument('--num_runs', type=int, default=4)
    return p.parse_args(args)

def seed_everything(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_output_path(model_name, dataset_name, dataset_path, max_length, hh=None, recent=None):
    base = "./time_profile_results"
    budget_dir = f"budget{hh + recent}" if (hh and recent) else "budget_full"
    model_dir_name = os.path.basename(model_name.rstrip("/"))
    sub_dir = os.path.basename(os.path.dirname(dataset_path)) if dataset_path and "evict_ruler" in dataset_path else ""
    output_dir = os.path.join(base, model_dir_name, budget_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    len_str = f"{int(max_length/1024)}k" if max_length % 1024 == 0 else f"{max_length}"
    return os.path.join(output_dir, f"{dataset_name}_{len_str}.jsonl")

CSV_HEADER = [
    "run_idx",
    "prefill_total(s)", "construct(s)", "prefill_writeback(s)", "prefill_attn(s)", "prefill_ffn(s)",
    "decode_avg(s)", "decode_writeback_avg(s)", "decode_attn_avg(s)", "retrieve_avg(s)", "decode_ffn_avg(s)",
]
REGISTRY_KEYS = [
    "run_idx",
    "prefill_total", "construct", "prefill_writeback", "prefill_attn", "prefill_ffn",
    "decode_avg", "decode_writeback_avg", "decode_attn_avg", "retrieve_avg", "decode_ffn_avg",
]

def collect_times(model, max_gen):
    construct = 0.0; prefill_writeback = 0.0; prefill_attn = 0.0
    decode_writeback_total = 0.0; decode_attn_total = 0.0; retrieve_total = 0.0

    for _name, module in model.named_modules():
        c = getattr(module, "kv_cache", None)
        if c is None or "self_attn" not in _name:
            continue
        c.construct_end_event.synchronize()
        construct += c.construct_start_event.elapsed_time(c.construct_end_event) / 1000.0

        c.prefill_writeback_end_event.synchronize()
        prefill_writeback += c.prefill_writeback_start_event.elapsed_time(c.prefill_writeback_end_event) / 1000.0

        c.prefill_attn_end_event.synchronize()
        prefill_attn += c.prefill_attn_start_event.elapsed_time(c.prefill_attn_end_event) / 1000.0

        decode_writeback_total += c.decode_writeback_event_time
        decode_attn_total += c.decode_attn_event_time
        retrieve_total += getattr(c, "retrieve_event_time", 0.0)

    return {
        "construct": construct, "prefill_writeback": prefill_writeback, "prefill_attn": prefill_attn,
        "decode_writeback_avg": decode_writeback_total / max_gen if max_gen > 0 else 0.0,
        "decode_attn_avg": decode_attn_total / max_gen if max_gen > 0 else 0.0,
        "retrieve_avg": retrieve_total / max_gen if max_gen > 0 else 0.0,
    }

def collect_global_times(max_gen):
    h2o_core.global_prefill_start_event.synchronize()
    h2o_core.global_prefill_end_event.synchronize()
    prefill_total = h2o_core.global_prefill_start_event.elapsed_time(
        h2o_core.global_prefill_end_event) / 1000.0
    h2o_core.global_decode_start_event.synchronize()
    h2o_core.global_decode_end_event.synchronize()
    decode_total = h2o_core.global_decode_start_event.elapsed_time(
        h2o_core.global_decode_end_event) / 1000.0
    decode_avg = decode_total / max_gen if max_gen > 0 else 0.0
    return {"prefill_total": prefill_total, "decode_avg": decode_avg}


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model_name = args.model; max_gen = 32
    dataset_path = args.dataset
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="flash_attention_2",
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    h2o_core.enable_h2o_time(model, heavy_hitter_size=args.heavy_hitter_size, recent_size=args.recent_size)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    input_ids_full = tokenizer(raw_text, return_tensors="pt").input_ids

    for max_length in [4096, 65536]:
        len_tag = "4k" if max_length == 4096 else "64k"
        if max_length > input_ids_full.shape[1]:
            print(f"Not enough tokens for {len_tag}, skipping.")
            continue

        output_path = build_output_path(model_name, dataset_name, dataset_path, max_length,
                                        args.heavy_hitter_size, args.recent_size)
        stats_csv = output_path.replace(".jsonl", "_breaktime.csv")
        print(f"\nInput Length: {len_tag} ({max_length} tokens)")

        h2o_core.breaktime_registry = []

        for run_idx in range(args.num_runs):
            print(f"   -> Run {run_idx + 1}/{args.num_runs}")
            h2o_core.reset_time_global_state(model)

            input_ids = input_ids_full[:, :max_length].to(model.device)
            context_length = input_ids.shape[-1]

            gen_kwargs = {
                "max_new_tokens": max_gen, "min_new_tokens": max_gen,
                "num_beams": 1, "do_sample": True,
                "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.15,
            }

            output = model.generate(input_ids=input_ids, **gen_kwargs)[0]
            h2o_core.global_decode_end_event.record()

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()

            local = collect_times(model, max_gen)
            global_ = collect_global_times(max_gen)

            prefill_ffn = global_["prefill_total"] - local["construct"] - local["prefill_writeback"] - local["prefill_attn"]
            decode_ffn_avg = global_["decode_avg"] - local["decode_writeback_avg"] - local["decode_attn_avg"] - local["retrieve_avg"]

            entry = {
                "run_idx": run_idx + 1,
                "prefill_total": global_["prefill_total"],
                "construct": local["construct"],
                "prefill_writeback": local["prefill_writeback"],
                "prefill_attn": local["prefill_attn"],
                "prefill_ffn": max(prefill_ffn, 0.0),
                "decode_avg": global_["decode_avg"],
                "decode_writeback_avg": local["decode_writeback_avg"],
                "decode_attn_avg": local["decode_attn_avg"],
                "retrieve_avg": local["retrieve_avg"],
                "decode_ffn_avg": max(decode_ffn_avg, 0.0),
            }
            h2o_core.breaktime_registry.append(entry)

            with open(output_path, "a", encoding="utf-8") as fout:
                json.dump({"run_idx": run_idx + 1, "pred": pred, "input_len": int(context_length)},
                          fout, ensure_ascii=False)
                fout.write('\n')
            torch.cuda.empty_cache()

        print(f"Saving CSV...")
        try:
            data = h2o_core.breaktime_registry
            with open(stats_csv, mode='w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                for item in data:
                    w.writerow([f"{item[k]:.6f}" for k in REGISTRY_KEYS])
                if len(data) >= 4:
                    r3, r4 = data[2], data[3]
                    avg = {"run_idx": 5}
                    for k in REGISTRY_KEYS[1:]:
                        avg[k] = (r3[k] + r4[k]) / 2
                    w.writerow([f"{avg[k]:.6f}" for k in REGISTRY_KEYS])
            print(f"{stats_csv}  ({len(data)} runs + avg)")
            last = data[-1]
            pf_sum = last["construct"] + last["prefill_writeback"] + last["prefill_attn"] + last["prefill_ffn"]
            de_sum = last["decode_writeback_avg"] + last["decode_attn_avg"] + last["retrieve_avg"] + last["decode_ffn_avg"]
            pf_cov = pf_sum / last["prefill_total"] * 100 if last["prefill_total"] > 0 else 0
            de_cov = de_sum / last["decode_avg"] * 100 if last["decode_avg"] > 0 else 0
            print(f"Prefill: {pf_cov:.1f}%  |  Decode: {de_cov:.1f}%")
        except Exception as e:
            print(f"CSV failed: {e}")
