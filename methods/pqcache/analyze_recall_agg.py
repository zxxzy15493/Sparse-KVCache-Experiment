#!/usr/bin/env python3
"""Analyze 1_all_mean.csv files under recall_list, sorted by model/dataset/budget/metric."""

import os
import re
import sys
import glob
import csv


def parse_folder_name(folder: str) -> dict:
    """Parse model, dataset, budget from folder name.
    New format: llama3.1-8b_fwe_bud1024
    Old format: llama-3.1_narrativeqa_budget_128_pq_search
    topk format: llama-3.1_narrativeqa_budget_128_no_drop_lb_topk
    """
    # New format: {model}_{dataset}_bud{budget} (optional suffix)
    m = re.match(r'^(.+?)_(.+?)_bud(\d+)(?:_.*)?$', folder)
    if m:
        return {'model': m.group(1), 'dataset': m.group(2), 'budget': int(m.group(3))}
    # Old format: {model}_{dataset}_budget_{budget} (optional suffix, e.g. _pq_search / _no_drop_lb_topk)
    m = re.match(r'^(.+?)_(.+?)_budget_(\d+)(?:_.*)?$', folder)
    if m:
        return {'model': m.group(1), 'dataset': m.group(2), 'budget': int(m.group(3))}

    return None


def model_order_key(model: str) -> int:
    """Prefix-match model order (llama < qwen)."""
    if model.startswith('llama'):
        return 0
    if model.startswith('qwen'):
        return 1
    return 99


def sort_key(info: dict):
    """Build sort key: model -> dataset -> metric -> budget."""
    dataset_order = {
        'niah_single_3': 0, 'vt': 1, 'fwe': 2,
        'narrativeqa': 0, 'qasper': 1,
    }
    metric_order = {'recall': 0, 'recall@100': 1, 'selected_attn': 2}

    return (
        model_order_key(info['model']),
        dataset_order.get(info['dataset'], 99),
        metric_order.get(info['metric'], 99),
        info['budget'],
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_recall_agg.py <base_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]

    pattern = os.path.join(base_dir, '*', 'analyze', '*_csv', '1_all_mean.csv')
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No 1_all_mean.csv found under {base_dir}")
        sys.exit(1)

    rows = []
    for csv_path in csv_files:
        # csv_path example:
        # .../llama-3.1_narrativeqa_budget_128_pq_search/analyze/llama-3.1_narrativeqa_budget_128_pq_search_20260522_130039_csv/1_all_mean.csv
        model_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(csv_path))))

        info = parse_folder_name(model_dir)
        if info is None:
            print(f"Skipping non-matching folder: {model_dir}")
            continue

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)

        for metric in ['recall', 'recall@100', 'selected_attn']:
            if metric not in row:
                continue
            info_with_metric = {**info, 'metric': metric}
            rows.append({
                'sort_key': sort_key(info_with_metric),
                'model': info['model'],
                'dataset': info['dataset'],
                'budget': info['budget'],
                'metric': metric,
                'value': float(row[metric]),
            })

    rows.sort(key=lambda x: x['sort_key'])

    print(f"{'Model':<20} {'Dataset':<15} {'Budget':<8} {'Metric':<15} {'Value'}")
    print("-" * 60)
    for r in rows:
        print(f"{r['model']:<20} {r['dataset']:<15} {r['budget']:<8} {r['metric']:<15} {r['value']:.6f}")

    out_csv = os.path.join(base_dir, 'analyze_recall_values.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([round(r['value'], 4) for r in rows])
    print(f"\nSaved to {out_csv}")


if __name__ == '__main__':
    main()
