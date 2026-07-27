import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def build_input_text(item):
  return (
    f"Context:\n{item.get('context', '')}\n\n"
    f"Question:\n{item.get('question', '')}\n\n"
    f"A. {item.get('choice_A', '')}\n"
    f"B. {item.get('choice_B', '')}\n"
    f"C. {item.get('choice_C', '')}\n"
    f"D. {item.get('choice_D', '')}\n"
  )


def get_token_len(tokenizer, text: str) -> int:
  ids = tokenizer(text, add_special_tokens=False)["input_ids"]
  return len(ids)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--tokenizer_name", type=str, required=True)
  parser.add_argument("--split", type=str, default="train")
  parser.add_argument("--min_len", type=int, default=0)
  parser.add_argument("--max_len", type=int, default=200000)
  parser.add_argument("--output_path", type=str, default="longbench_v2_fullinput_lt_200k.jsonl")
  parser.add_argument("--dataset_name", type=str, default="THUDM/LongBench-v2")
  args = parser.parse_args()

  if args.min_len < 0:
    raise ValueError("--min_len must be >= 0")
  if args.max_len <= args.min_len:
    raise ValueError("--max_len must be greater than --min_len")

  tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name,
    trust_remote_code=True
  )
  dataset = load_dataset(args.dataset_name, split=args.split)

  kept = 0
  total = 0

  os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

  with open(args.output_path, "w", encoding="utf-8") as fout:
    for item in tqdm(dataset, desc="Filtering"):
      total += 1
      full_text = build_input_text(item)
      seq_len = get_token_len(tokenizer, full_text)

      if args.min_len <= seq_len < args.max_len:
        item = dict(item)
        item["tokenized_input_len"] = seq_len
        item["tokenized_input_len_k"] = f"{seq_len / 1024:.1f}k"
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        kept += 1

  print(f"Total samples: {total}")
  print(f"Kept samples : {kept}")
  print(f"Length range : [{args.min_len}, {args.max_len}) tokens")
  print(f"Saved to   : {args.output_path}")


if __name__ == "__main__":
  main()
