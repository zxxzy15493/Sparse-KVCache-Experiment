#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import Any, Dict, List, Set, Tuple

Key = Tuple[int, int]  # (index, layer)

def flatten_ints(obj: Any, out: List[int]) -> None:
    """ int  out"""
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


def to_set(rec: Dict[str, Any], field: str) -> Set[int]:
    if field not in rec:
        return set()
    vals: List[int] = []
    flatten_ints(rec[field], vals)
    return set(vals)

def safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b else 0.0

def load_index_layer_map(path: str, field: str) -> Dict[Key, Set[int]]:
    mp: Dict[Key, Set[int]] = {}
    dup = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)

            if "index" not in rec or "layer" not in rec:
                continue

            k: Key = (int(rec["index"]), int(rec["layer"]))
            s = to_set(rec, field)

            if k in mp:

                mp[k] |= s
                dup += 1
            else:
                mp[k] = s
    return mp

def main():
    ap = argparse.ArgumentParser(description="Compare token_position by matching (index, layer) between two jsonl files.")
    ap.add_argument("jsonl_a", type=str, help="file A")
    ap.add_argument("jsonl_b", type=str, help="file B")
    ap.add_argument("--field", type=str, default="token_position", help="field name (default: token_position)")
    ap.add_argument("--out_jsonl", type=str, default="compare_by_index_layer.jsonl", help="output jsonl")
    args = ap.parse_args()


    map_a = load_index_layer_map(args.jsonl_a, args.field)


    matched = 0
    sum_inter = 0
    sum_union = 0
    sum_size_a = 0
    sum_size_b = 0
    global_a: Set[int] = set()
    global_b: Set[int] = set()

    keys_b: Set[Key] = set()


    with open(args.jsonl_b, "r", encoding="utf-8") as fb, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fb:
            line = line.strip()
            if not line:
                continue
            rec_b = json.loads(line)

            if "index" not in rec_b or "layer" not in rec_b:
                continue

            k: Key = (int(rec_b["index"]), int(rec_b["layer"]))
            keys_b.add(k)

            if k not in map_a:
                continue

            A = map_a[k]
            B = to_set(rec_b, args.field)

            inter = len(A & B)
            union = len(A | B)
            size_a = len(A)
            size_b = len(B)

            out = {
                "index": k[0],
                "layer": k[1],
                "A_size": size_a,
                "B_size": size_b,
                "intersection": inter,
                "union": union,
                "jaccard": safe_div(inter, union),
                "precision_A_in_B": safe_div(inter, size_a),
                "recall_A_cover_B": safe_div(inter, size_b),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            matched += 1
            sum_inter += inter
            sum_union += union
            sum_size_a += size_a
            sum_size_b += size_b
            global_a |= A
            global_b |= B

    keys_a = set(map_a.keys())
    only_a = len(keys_a - keys_b)
    only_b = len(keys_b - keys_a)

    g_inter = len(global_a & global_b)
    g_union = len(global_a | global_b)

    print("==== Key matching summary (by index, layer) ====")
    print(f"A_keys={len(keys_a)}, B_keys={len(keys_b)}, matched_keys={matched}")
    print(f"only_in_A={only_a}, only_in_B={only_b}")
    print(f"output written to: {args.out_jsonl}")

    print("\n==== Micro over matched keys (sum over pairs) ====")
    print(f"sum_A_size={sum_size_a}, sum_B_size={sum_size_b}")
    print(f"sum_intersection={sum_inter}, sum_union={sum_union}")
    print(f"micro_jaccard(sum_inter/sum_union)={safe_div(sum_inter, sum_union):.6f}")
    print(f"micro_precision_A_in_B(sum_inter/sum_A_size)={safe_div(sum_inter, sum_size_a):.6f}")
    print(f"micro_recall_A_cover_B(sum_inter/sum_B_size)={safe_div(sum_inter, sum_size_b):.6f}")

    print("\n==== Global unique-token set level (matched keys only) ====")
    print(f"global_A_unique={len(global_a)}, global_B_unique={len(global_b)}")
    print(f"global_intersection={g_inter}, global_union={g_union}")
    print(f"global_jaccard={safe_div(g_inter, g_union):.6f}")
    print(f"global_precision_A_in_B={safe_div(g_inter, len(global_a)):.6f}")
    print(f"global_recall_A_cover_B={safe_div(g_inter, len(global_b)):.6f}")

if __name__ == "__main__":
    main()
