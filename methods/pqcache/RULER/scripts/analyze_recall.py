#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze recall logs under:
    recall_list/<RECALL_NAME>/<RECALL_NAME>*.csv

Expected log format:
    # ===== PROMPT 1 | 2026-05-18 16:20:31 =====
    # prefill_len: 8192
    layer,head,recall,recall@100,selected_attn
      0,  0, 0.7123, 0.6500, 0.0004
      0,  1, 0.6981, 0.6200, 0.0003
      ...

Also compatible with old format:
    layer,head,recall,selected_attn

The script infers decode token position inside each prompt by row order:
    when layer index decreases, token_pos += 1

Outputs:
    recall_list/<RECALL_NAME>/analyze/<input_stem>_csv/
        raw_with_token_pos.csv
        1_all_mean.csv
        2_each_prompt_mean.csv
        3_prompt_token_mean.csv
        4_prompt_token_layer.csv
        5_layer_mean.csv
        6_layer_head_mean.csv

    recall_list/<RECALL_NAME>/analyze/<input_stem>_analyze_token_curve.png
"""

import argparse
import re
from pathlib import Path

import pandas as pd


PROMPT_RE = re.compile(r"#\s*=+\s*PROMPT\s+(\d+)\s*\|?(.*?)=*", re.IGNORECASE)
PREFILL_RE = re.compile(r"#\s*prefill_len\s*:\s*(\d+)", re.IGNORECASE)


def parse_recall_file(path: Path) -> pd.DataFrame:
    rows = []
    prompt_id = 0
    prefill_len = None
    prompt_time = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                m = PROMPT_RE.search(line)
                if m:
                    prompt_id = int(m.group(1))
                    prompt_time = m.group(2).strip(" =|")
                    prefill_len = None
                    continue

                m = PREFILL_RE.search(line)
                if m:
                    prefill_len = int(m.group(1))
                    continue

                continue

            if line.lower().startswith("layer,head,"):
                continue

            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4:
                continue

            if prompt_id == 0:
                prompt_id = 1

            try:
                layer = int(parts[0])
                head = int(parts[1])
                recall = float(parts[2])

                # New format:
                #   layer,head,recall,recall@100,selected_attn
                # Old format:
                #   layer,head,recall,selected_attn
                if len(parts) >= 5:
                    recall_100 = float(parts[3])
                    selected_attn = float(parts[4])
                else:
                    recall_100 = None
                    selected_attn = float(parts[3])
            except ValueError:
                continue

            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_time": prompt_time,
                    "prefill_len": prefill_len,
                    "line_no": line_no,
                    "layer": layer,
                    "head": head,
                    "recall": recall,
                    "recall@100": recall_100,
                    "selected_attn": selected_attn,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = infer_decode_token_pos(df)
    return df


def infer_decode_token_pos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["prompt_id", "line_no"]).copy()
    token_positions = []

    for _, g in df.groupby("prompt_id", sort=False):
        token_pos = 0
        prev_layer = None

        for layer in g["layer"].tolist():
            if prev_layer is not None and layer < prev_layer:
                token_pos += 1

            token_positions.append(token_pos)
            prev_layer = layer

    df["token_pos"] = token_positions
    return df


def summarize(df: pd.DataFrame):
    """
    Returns the six requested summaries.

    1. all data, all decode tokens, all layers, all heads
    2. each data/prompt, all decode tokens, all layers, all heads
    3. each data/prompt, each decode token, all layers, all heads
    4. each data/prompt, each decode token, each layer, all heads
    5. all data, all decode tokens, each layer, all heads
    6. all data, all decode tokens, each layer, each head
    """
    metrics = ["recall", "recall@100", "selected_attn"]

    summary_all = (
        df[metrics]
        .mean()
        .to_frame("mean")
        .T
        .reset_index(drop=True)
    )

    by_prompt = (
        df.groupby("prompt_id", as_index=False)[metrics]
        .mean()
        .sort_values("prompt_id")
    )

    by_prompt_token = (
        df.groupby(["prompt_id", "token_pos"], as_index=False)[metrics]
        .mean()
        .sort_values(["prompt_id", "token_pos"])
    )

    by_prompt_token_layer = (
        df.groupby(["prompt_id", "token_pos", "layer"], as_index=False)[metrics]
        .mean()
        .sort_values(["prompt_id", "token_pos", "layer"])
    )

    by_layer = (
        df.groupby("layer", as_index=False)[metrics]
        .mean()
        .sort_values("layer")
    )

    by_layer_head = (
        df.groupby(["layer", "head"], as_index=False)[metrics]
        .mean()
        .sort_values(["layer", "head"])
    )

    return {
        "all_mean": summary_all,
        "prompt_mean": by_prompt,
        "prompt_token_mean": by_prompt_token,
        "prompt_token_layer_mean": by_prompt_token_layer,
        "layer_mean": by_layer,
        "layer_head_mean": by_layer_head,
    }


def save_csvs(out_dir: Path, input_stem: str, raw_df: pd.DataFrame, tables: dict) -> Path:
    csv_dir = out_dir / f"{input_stem}_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(csv_dir / "raw_with_token_pos.csv", index=False)

    file_names = {
        "all_mean": "1_all_mean.csv",
        "prompt_mean": "2_each_prompt_mean.csv",
        "prompt_token_mean": "3_prompt_token_mean.csv",
        "prompt_token_layer_mean": "4_prompt_token_layer.csv",
        "layer_mean": "5_layer_mean.csv",
        "layer_head_mean": "6_layer_head_mean.csv",
    }

    for key, table in tables.items():
        table.to_csv(csv_dir / file_names[key], index=False)

    return csv_dir


def save_token_curve_png(out_png: Path, by_prompt_token: pd.DataFrame):
    import matplotlib.pyplot as plt

    prompts = sorted(by_prompt_token["prompt_id"].unique())

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for prompt_id in prompts:
        g = by_prompt_token[by_prompt_token["prompt_id"] == prompt_id]

        axes[0].plot(
            g["token_pos"],
            g["recall"],
            marker="o",
            label=f"prompt {prompt_id}",
        )

        axes[1].plot(
            g["token_pos"],
            g["recall@100"],
            marker="o",
            label=f"prompt {prompt_id}",
        )

        axes[2].plot(
            g["token_pos"],
            g["selected_attn"],
            marker="o",
            label=f"prompt {prompt_id}",
        )

    axes[0].set_ylabel("mean recall")
    axes[0].set_title("Mean recall per decode token")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_ylabel("mean recall@100")
    axes[1].set_title("Mean recall@100 per decode token")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_xlabel("decode token position")
    axes[2].set_ylabel("mean selected_attn")
    axes[2].set_title("Mean selected_attn per decode token")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def analyze_one_file(path: Path, out_dir: Path):
    df = parse_recall_file(path)
    if df.empty:
        print(f"[skip] no valid data: {path}")
        return

    tables = summarize(df)

    out_csv_dir = save_csvs(out_dir, path.stem, df, tables)
    out_png = out_dir / f"{path.stem}_analyze_token_curve.png"

    save_token_curve_png(out_png, tables["prompt_token_mean"])

    print(f"[ok] {path.name}")
    print(f"     -> {out_csv_dir}")
    print(f"     -> {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recall-name",
        required=True,
        help="RECALL_NAME, e.g. qwen_pq. Logs are read from recall_list/<RECALL_NAME>/",
    )
    parser.add_argument(
        "--root",
        default="recall_list",
        help="Root directory. Default: recall_list",
    )
    args = parser.parse_args()

    root = Path(args.root)
    recall_dir = root / args.recall_name
    out_dir = recall_dir / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not recall_dir.exists():
        raise FileNotFoundError(f"Directory not found: {recall_dir}")

    files = sorted(
        p for p in recall_dir.glob(f"{args.recall_name}*.csv")
        if "_analyze" not in p.stem
    )

    if not files:
        raise FileNotFoundError(
            f"No files matched: {recall_dir}/{args.recall_name}*.csv"
        )

    for path in files:
        analyze_one_file(path, out_dir)


if __name__ == "__main__":
    main()