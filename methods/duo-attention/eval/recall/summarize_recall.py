#!/usr/bin/env python3
"""Scan all _attn_ratios.csv under recall_results/ and print averages."""

import pandas as pd
import os, sys, glob

root = sys.argv[1] if len(sys.argv) > 1 else "recall_results"

rows = []
for path in sorted(glob.glob(os.path.join(root, "**", "*_attn_ratios.csv"), recursive=True)):
    df = pd.read_csv(path)
    rel = os.path.relpath(path, root)
    comps = rel.split("/")
    # Structure variants:
    #   <model>/sp<N>/<category>/<dataset>_attn_ratios.csv        (DuoAttention, 4-part)
    #   <method>/<model>/budget<N>/<dataset>_attn_ratios.csv      (HeadKV, 4-part)
    #   <model>/budget<N>/<dataset>_attn_ratios.csv               (CakeKV, 3-part)
    #
    # The dataset name is always derived from the filename (last element).
    dataset = comps[-1].replace("_attn_ratios.csv", "")
    if len(comps) == 3:
        method, model, budget_dir = "", comps[0], comps[1]
    elif len(comps) == 4:
        if comps[1].startswith("sp"):
            method, model, budget_dir = "", comps[0], comps[1]
        else:
            method, model, budget_dir = comps[0], comps[1], comps[2]
    elif len(comps) == 5:
        method, model, budget_dir = comps[0], comps[1], comps[2]
    else:
        continue
    budget = budget_dir.replace("sp", "").replace("budget", "")

    rows.append({
        "method": method or "-",
        "model": model,
        "budget": int(budget),
        "dataset": dataset,
        "samples": df["sample_idx"].nunique(),
        "layers": df["layer_idx"].nunique(),
        "recall_100": df["recall_100"].mean(),
        "recall_k": df["recall_k"].mean(),
        "selected_attn_ratio": df["selected_attn_ratio"].mean(),
    })

if not rows:
    print("No CSV files found under", root)
    sys.exit(1)

summary = pd.DataFrame(rows)
# Custom dataset order: na (narrativeqa), qa (qasper), ns3 (niah_single_3), vt, cwe
_ds_order = {"narrativeqa": 0, "qasper": 1, "niah_single_3": 2, "vt": 3, "cwe": 4}
summary["_ds_order"] = summary["dataset"].map(_ds_order).fillna(99).astype(int)
summary = summary.sort_values(["method", "model", "_ds_order", "budget"]).drop(columns="_ds_order")

print(f"{'Method':<12} {'Model':<30} {'Budget':>6} {'Dataset':<20} {'Recall@100':>10} {'Recall@k':>10} {'AttnRatio':>10} {'#Samples':>9}")
print("-" * 107)
for _, r in summary.iterrows():
    print(f"{r['method']:<12} {r['model']:<30} {r['budget']:>6} {r['dataset']:<20} {r['recall_100']:>10.4f} {r['recall_k']:>10.4f} {r['selected_attn_ratio']:>10.4f} {r['samples']:>9}")

# Also print overall mean by method+model+dataset
print("\n--- Mean by (method, model, dataset) across budgets ---")
gb = summary.groupby(["method", "model", "dataset"])[["recall_100", "recall_k", "selected_attn_ratio"]].mean().reset_index()
gb["_ds_order"] = gb["dataset"].map(_ds_order).fillna(99).astype(int)
gb = gb.sort_values(["method", "model", "_ds_order"]).drop(columns="_ds_order")
print(f"{'Method':<12} {'Model':<30} {'Dataset':<20} {'Recall@100':>10} {'Recall@k':>10} {'AttnRatio':>10}")
print("-" * 82)
for _, r in gb.iterrows():
    print(f"{r['method']:<12} {r['model']:<30} {r['dataset']:<20} {r['recall_100']:>10.4f} {r['recall_k']:>10.4f} {r['selected_attn_ratio']:>10.4f}")