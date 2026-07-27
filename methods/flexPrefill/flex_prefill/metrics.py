"""
FlexPrefill sparse attention quality metrics plugin.

Encapsulates computation and result parsing for recall and captured_mass (top-k rate).
"""

from __future__ import annotations

import gc
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flex_prefill import patch_model


_DEFAULT_EFFICIENCY_DIR = Path("efficiency")


def _resolve_output_dir() -> Path:
  env = os.environ.get("FLEXPREFILL_EFFICIENCY_DIR", "")
  if env:
    return Path(env)
  return _DEFAULT_EFFICIENCY_DIR


def parse_recall_results(model_name: str, gamma: float, tau: float,
             block_size: int = 128, min_budget: int = 1024,
             results_dir: Optional[Path] = None) -> pd.DataFrame:
  """Parse recall result JSONL into a DataFrame."""
  if results_dir is None:
    results_dir = _resolve_output_dir() / "recall-results"
  path = results_dir / f"{model_name}-block_size{block_size}-min_budget{min_budget}-gamma{gamma}-tau{tau}.jsonl"
  return _parse_jsonl(path, metric_col="avg_recall_pre_head")


def parse_captured_mass_results(model_name: str, gamma: float, tau: float,
                block_size: int = 128, min_budget: int = 1024,
                results_dir: Optional[Path] = None) -> pd.DataFrame:
  """Parse captured_mass result JSONL into a DataFrame."""
  if results_dir is None:
    results_dir = _resolve_output_dir() / "captured_mass-results"
  path = results_dir / f"{model_name}-block_size{block_size}-min_budget{min_budget}-gamma{gamma}-tau{tau}.jsonl"
  return _parse_jsonl(path, metric_col="avg_captured_mass_pre_head")


def _parse_jsonl(path: Path, metric_col: str) -> pd.DataFrame:
  if not path.exists():
    raise FileNotFoundError(f"Results file not found: {path}")
  records = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  return pd.DataFrame(records)


def summarize(df: pd.DataFrame, metric_col: str) -> Dict:
  """Compute summary statistics over a metric DataFrame.

  Returns:
    overall_mean, overall_std,
    per_layer_mean (dict), per_head_type_mean (dict),
    sample_count
  """
  overall_mean = float(df[metric_col].mean())
  overall_std = float(df[metric_col].std())

  per_layer = df.groupby("layer")[metric_col].mean().to_dict()
  per_layer = {int(k): float(v) for k, v in per_layer.items()}

  per_head_type = df.groupby("head_type")[metric_col].mean().to_dict()
  per_head_type = {k: float(v) for k, v in per_head_type.items()}

  return {
    "overall_mean": overall_mean,
    "overall_std": overall_std,
    "per_layer_mean": per_layer,
    "per_head_type_mean": per_head_type,
    "sample_count": len(df),
  }


class FlexPrefillMetrics:
  """FlexPrefill attention quality metrics calculator.

  Both recall and captured_mass metrics are computed automatically during
  the flex_prefill_attention forward pass and written to JSONL files.
  This class wraps: model loading + flex_prefill patch, prefill forward
  on arbitrary prompts, result parsing to DataFrames, and aggregation.

  Note: full attention weights require O(n^2) memory, so seqlen <= 16k
  is recommended and batch_size is fixed to 1.
  """

  def __init__(
    self,
    model_path: str,
    gamma: float = 0.9,
    tau: float = 0.1,
    block_size: int = 128,
    min_budget: int = 1024,
    max_budget: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    results_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
  ):
    self.model_path = model_path
    self.gamma = gamma
    self.tau = tau
    self.block_size = block_size
    self.min_budget = min_budget
    self.max_budget = max_budget
    self.results_dir = results_dir

    if device is None:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device(device)

    if model_name is None:
      name = Path(model_path).name
      if "Llama" in name:
        self.model_name = "Llama"
      elif "Qwen" in name:
        self.model_name = "Qwen"
      else:
        self.model_name = name
    else:
      self.model_name = model_name

    self._model = None
    self._tokenizer = None

  def load_model(self):
    """Load and patch the model (idempotent)."""
    if self._model is not None:
      return

    self._tokenizer = AutoTokenizer.from_pretrained(
      self.model_path, trust_remote_code=True
    )
    self._model = AutoModelForCausalLM.from_pretrained(
      self.model_path,
      torch_dtype=torch.bfloat16,
      device_map="cuda",
      _attn_implementation="flash_attention_2",
      trust_remote_code=True,
    )

    config = {
      "block_size": self.block_size,
      "flex_prefill_gamma": self.gamma,
      "flex_prefill_tau": self.tau,
      "flex_prefill_min_budget": self.min_budget,
      "flex_prefill_max_budget": self.max_budget,
    }
    patch_model(self._model, "flex_prefill", config)
    self._model.eval()

  def unload_model(self):
    """Release model memory."""
    if self._model is not None:
      del self._model
      self._model = None
    if self._tokenizer is not None:
      del self._tokenizer
      self._tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

  @property
  def model(self):
    if self._model is None:
      self.load_model()
    return self._model

  @property
  def tokenizer(self):
    if self._tokenizer is None:
      self.load_model()
    return self._tokenizer

  def _result_suffix(self) -> str:
    return f"{self.model_name}-block_size{self.block_size}-min_budget{self.min_budget}-gamma{self.gamma}-tau{self.tau}"

  def _outpath_for(self, metric_type: str) -> Path:
    """Internal metrics file path (matches flex_prefill_attention.py output)."""
    if self.results_dir is not None:
      base = self.results_dir
    else:
      base = _resolve_output_dir()
    subdir = f"{metric_type}-results"
    return base / subdir / f"{self._result_suffix()}.jsonl"

  def _run_forward(self, prompts: List[str], seqlen: int, metric_type: str):
    """Run one prefill forward per prompt to trigger internal metric computation.

    Results are written to JSONL files by flex_prefill_attention internally;
    this function does not extract metrics from forward return values.
    """
    self.load_model()
    self.model.config.type = metric_type

    outpath = self._outpath_for(metric_type)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists():
      outpath.unlink()

    for prompt in prompts:
      inputs = self.tokenizer(
        prompt,
        truncation=True,
        max_length=seqlen,
        return_tensors="pt",
      ).to(self.device)

      with torch.no_grad():
        self.model(**inputs, use_cache=True)

      del inputs
      torch.cuda.empty_cache()

  def compute_recall(
    self, prompts: Union[str, List[str]], seqlen: int = 4096
  ) -> pd.DataFrame:
    """Compute recall and return a DataFrame.

    recall: for each (head, query_pos), the intersection ratio between
    FlexPrefill's selected token set and the true full-attention top-K set.

    Returns:
      DataFrame with columns: layer, head_num, avg_recall_pre_head,
      head_type, q_len, block_num
    """
    if isinstance(prompts, str):
      prompts = [prompts]
    self._run_forward(prompts, seqlen, "recall")
    return parse_recall_results(
      self.model_name, self.gamma, self.tau,
      self.block_size, self.min_budget,
      results_dir=self.results_dir,
    )

  def compute_captured_mass(
    self, prompts: Union[str, List[str]], seqlen: int = 4096
  ) -> pd.DataFrame:
    """Compute captured_mass (top-k rate) and return a DataFrame.

    captured_mass: for each (head, query_pos), the attention probability
    mass captured by FlexPrefill's selected blocks (equivalent to top-k rate).

    Returns:
      DataFrame with columns: layer, head_num, avg_captured_mass_pre_head,
      head_type, q_len, block_num
    """
    if isinstance(prompts, str):
      prompts = [prompts]
    self._run_forward(prompts, seqlen, "captured_mass")
    return parse_captured_mass_results(
      self.model_name, self.gamma, self.tau,
      self.block_size, self.min_budget,
      results_dir=self.results_dir,
    )

  def compute_both(
    self, prompts: Union[str, List[str]], seqlen: int = 4096
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute both recall and captured_mass in one call.

    Requires two independent forward passes (one per metric type).
    """
    if isinstance(prompts, str):
      prompts = [prompts]
    df_recall = self.compute_recall(prompts, seqlen)
    df_captured = self.compute_captured_mass(prompts, seqlen)
    return df_recall, df_captured

  def print_summary(self, metric: str, df: Optional[pd.DataFrame] = None):
    """Print a summary table for the given metric.

    Args:
      metric: "recall" or "captured_mass"
      df: Optional pre-computed DataFrame. Reads from file if not provided.
    """
    if df is None:
      if metric == "recall":
        df = parse_recall_results(
          self.model_name, self.gamma, self.tau,
          self.block_size, self.min_budget,
          results_dir=self.results_dir,
        )
      else:
        df = parse_captured_mass_results(
          self.model_name, self.gamma, self.tau,
          self.block_size, self.min_budget,
          results_dir=self.results_dir,
        )

    metric_col = {
      "recall": "avg_recall_pre_head",
      "captured_mass": "avg_captured_mass_pre_head",
    }[metric]

    stats = summarize(df, metric_col)

    print(f"\n{'='*60}")
    print(f" {metric.upper()} Summary")
    print(f" model={self.model_name}, gamma={self.gamma}, tau={self.tau}")
    print(f" block_size={self.block_size}, min_budget={self.min_budget}")
    print(f" samples (heads): {stats['sample_count']}")
    print(f"{'='*60}")
    print(f" Overall mean : {stats['overall_mean']:.4f}")
    print(f" Overall std : {stats['overall_std']:.4f}")
    print(f" Per head type:")
    for ht, v in stats["per_head_type_mean"].items():
      print(f"  {ht:20s}: {v:.4f}")
    print(f" Per layer (first 5):")
    for layer, v in list(stats["per_layer_mean"].items())[:5]:
      print(f"  layer {layer:2d}: {v:.4f}")
    print(f"{'='*60}\n")

  def scan_gamma(
    self,
    prompts: Union[str, List[str]],
    gammas: List[float],
    seqlen: int = 4096,
    metric: str = "recall",
  ) -> pd.DataFrame:
    """Scan across different gamma values, returns a summary DataFrame."""
    if isinstance(prompts, str):
      prompts = [prompts]

    rows = []
    for g in gammas:
      self.gamma = g

      self.unload_model()
      self.load_model()

      if metric == "recall":
        df = self.compute_recall(prompts, seqlen)
        metric_col = "avg_recall_pre_head"
      else:
        df = self.compute_captured_mass(prompts, seqlen)
        metric_col = "avg_captured_mass_pre_head"

      stats = summarize(df, metric_col)
      row = {
        "gamma": g,
        "overall_mean": stats["overall_mean"],
        "overall_std": stats["overall_std"],
      }
      for ht, v in stats["per_head_type_mean"].items():
        row[f"{ht}_mean"] = v
      rows.append(row)

    return pd.DataFrame(rows)

  def scan_seqlen(
    self,
    prompts: Union[str, List[str]],
    seqlens: List[int],
    metric: str = "recall",
  ) -> pd.DataFrame:
    """Scan across different sequence lengths, returns a summary DataFrame."""
    if isinstance(prompts, str):
      prompts = [prompts]

    rows = []
    for sl in seqlens:
      if metric == "recall":
        df = self.compute_recall(prompts, sl)
        metric_col = "avg_recall_pre_head"
      else:
        df = self.compute_captured_mass(prompts, sl)
        metric_col = "avg_captured_mass_pre_head"

      stats = summarize(df, metric_col)
      row = {
        "seqlen": sl,
        "overall_mean": stats["overall_mean"],
        "overall_std": stats["overall_std"],
      }
      for ht, v in stats["per_head_type_mean"].items():
        row[f"{ht}_mean"] = v
      rows.append(row)

    return pd.DataFrame(rows)
