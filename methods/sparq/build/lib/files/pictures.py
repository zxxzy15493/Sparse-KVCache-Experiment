#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import Any, List, Dict

import numpy as np
import matplotlib.pyplot as plt


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



def parse_layers_arg(layers_str: str) -> List[int]:

    return [int(x.strip()) for x in layers_str.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description="Plot per-layer token_position histogram for first 5 layers.")
    ap.add_argument("input_jsonl", type=str, help="input jsonl path")
    ap.add_argument("--field", type=str, default="token_position", help="field name (default: token_position)")
    ap.add_argument("--layers", type=str, default="0,1,2,3,4", help="layers to plot, e.g. '0,1,2,3,4'")
    ap.add_argument("--bins", type=int, default=200, help="number of histogram bins (default: 200)")
    ap.add_argument("--out_prefix", type=str, default="tokenpos", help="output prefix (default: tokenpos)")
    ap.add_argument("--log_y", action="store_true", help="use log scale on y axis")
    args = ap.parse_args()

    layers = parse_layers_arg(args.layers)
    layer_set = set(layers)


    max_pos: Dict[int, int] = {l: -1 for l in layers}
    used_lines = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if "layer" not in rec or args.field not in rec:
                continue
            layer = int(rec["layer"])
            if layer not in layer_set:
                continue

            vals: List[int] = []
            flatten_ints(rec[args.field], vals)
            if not vals:
                continue

            m = max(vals)
            if m > max_pos[layer]:
                max_pos[layer] = m
            used_lines += 1


    valid_layers = [l for l in layers if max_pos[l] >= 0]
    if not valid_layers:
        raise RuntimeError(f" layers={layers}  {args.field} ")


    counts: Dict[int, np.ndarray] = {l: np.zeros(args.bins, dtype=np.int64) for l in valid_layers}

    bin_width: Dict[int, float] = {}
    for l in valid_layers:

        bin_width[l] = (max_pos[l] + 1) / float(args.bins)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if "layer" not in rec or args.field not in rec:
                continue
            layer = int(rec["layer"])
            if layer not in counts:
                continue

            vals: List[int] = []
            flatten_ints(rec[args.field], vals)
            if not vals:
                continue

            bw = bin_width[layer]

            for pos in vals:
                idx = int(pos / bw) if bw > 0 else 0
                if idx < 0:
                    idx = 0
                elif idx >= args.bins:
                    idx = args.bins - 1
                counts[layer][idx] += 1


    for layer in valid_layers:
        bw = bin_width[layer]

        centers = (np.arange(args.bins) + 0.5) * bw

        plt.figure()
        plt.bar(centers, counts[layer], width=bw * 0.95)
        if args.log_y:
            plt.yscale("log")

        plt.xlabel("token_position")
        plt.ylabel("count")
        plt.title(f"token_position histogram | layer={layer} | bins={args.bins} | max_pos={max_pos[layer]}")
        out_path = f"{args.out_prefix}_layer{layer}_hist_tokenpos.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("saved:", out_path)

    print("done.")


if __name__ == "__main__":
    main()
