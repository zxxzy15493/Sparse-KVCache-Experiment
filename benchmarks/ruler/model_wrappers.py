from __future__ import annotations

import math
import random
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from loaders import load_model_and_tokenizer


def seed_everything() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class RulerModel:
    """Local unified-model wrapper used by the RULER call_api entrypoint."""

    def __init__(self, method: str, model: str, budget: int, device: str, overrides: list[str], max_seq_length: int | None = None):
        seed_everything()
        self.method = method
        self.args = Namespace(method=method, model=model, budget=budget, device=device, set=overrides, max_seq_length=max_seq_length)
        ## 打印args的所有值
        print("RulerModel args: =================================== ================================")
        for key, value in vars(self.args).items():
            print(f"{key}: {value}")
        self.model, self.tokenizer = load_model_and_tokenizer(self.args)

    def _prepare_clusterkv(self, context_length: int) -> None:
        if self.method != "clusterkv":
            return
        cluster_args = self.args._method_args
        cluster_args.nlist = max(self.args._base_nlist, math.ceil(context_length / 80))
        for module in self.model.modules():
            if hasattr(module, "flash_forward"):
                self.args._cluster_attention.apply_cluster_config(module, cluster_args)
                module.token_budget = cluster_args.token_budget
                module.cluster_cache = None
        self.args._cluster_attention.cluster_reset(self.model)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.model.device)
        context_length = inputs.input_ids.shape[-1]
        self._prepare_clusterkv(context_length)
        try:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            return self.tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()
        finally:
            if self.method == "clusterkv":
                self.args._cluster_attention.cluster_reset(self.model)
            torch.cuda.empty_cache()

    def close(self) -> None:
        if getattr(self.args, "_cleanup", None):
            self.args._cleanup()
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
