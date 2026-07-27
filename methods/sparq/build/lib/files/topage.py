#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import Counter
from typing import Any, List

def map_token_pos_to_page(obj: Any, page_size: int, page_one_based: bool) -> Any:
    """token position(int) -> page id(int)"""
    if obj is None:
        return None
    if isinstance(obj, int):
        page = obj // page_size
        return page + 1 if page_one_based else page
    if isinstance(obj, list):
        return [map_token_pos_to_page(x, page_size, page_one_based) for x in obj]
    if isinstance(obj, dict):

        if "idx" in obj and isinstance(obj["idx"], int):
            page = obj["idx"] // page_size
            out = dict(obj)
            out["page"] = page + 1 if page_one_based else page
            return out
        return {k: map_token_pos_to_page(v, page_size, page_one_based) for k, v in obj.items()}
    return obj

def flatten_ints(obj: Any, out: List[int]) -> None:
    """ int  out int"""
    if obj is None:
        return
    if isinstance(obj, int):
        out.append(obj)
        return
    if isinstance(obj, list):
        for x in obj:
            flatten_ints(x, out)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            flatten_ints(v, out)
        return


def main():
    parser = argparse.ArgumentParser(description="Compute page ids for token positions in jsonl and sort them.")
    parser.add_argument("input_jsonl", type=str, help="input jsonl path")
    parser.add_argument("output_jsonl", type=str, help="output jsonl path")
    parser.add_argument("--page_size", type=int, default=16, help="page size (default: 16)")
    parser.add_argument("--field", type=str, default="token_position", help="field name containing token positions")
    parser.add_argument("--page_one_based", action="store_true", help="use 1-based page id (default: 0-based)")
    args = parser.parse_args()

    total, missing = 0, 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            rec = json.loads(line)
            if args.field not in rec:
                missing += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            token_pos = rec[args.field]


            pages_nested = map_token_pos_to_page(token_pos, args.page_size, args.page_one_based)


            pages_flat: List[int] = []
            flatten_ints(pages_nested, pages_flat)
            pages_sorted = sorted(pages_flat)


            freq = Counter(pages_flat)
            page_freq_sorted = [
                {"page": int(p), "count": int(c)}
                for p, c in sorted(freq.items(), key=lambda x: (-x[1], x[0]))
            ]

            #rec["token_page_nested"] = pages_nested
            rec["token_page_sorted"] = pages_sorted
            #rec["page_freq_sorted"] = page_freq_sorted
            rec.pop(args.field, None)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. total_lines={total}, missing_field={missing}")
    print(f"Output: {args.output_jsonl}")

if __name__ == "__main__":
    main()
