#
#
#
from lm_eval.models.huggingface import HFLM
import torch
import transformers
from typing import List, Tuple
import random
import numpy as np
import torch.nn.functional as F
from functools import partial
import argparse
from ast import literal_eval
from tqdm import tqdm
import copy
from time import time
from lm_eval.api.instance import Instance
from lm_eval.models.utils import Collator


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="model path",
  )
  parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="tasks to evaluate on, use comma to split different tasks",
  )
  parser.add_argument(
    "--save_dir",
    type=str,
    default="experiments/result",
    help="save path, default to experiments/result/",
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=-1,
    help="Eval samples limit, only used for debug. Set -1 to run full eval.",
  )
  parser.add_argument(
    "--max_length",
    type=int,
    default=131072,
    help="Max length of input, default 128k.",
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="batch size, set -1 if want to use auto batch size.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed",
  )
  parser.add_argument(
    "--top_p",
    type=float,
    default=-1,
    help="top p, default to 0.7, set -1 to use greedy decoding",
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=1,
    help="temperature, default to 1",
  )
  parser.add_argument(
    "--chat",
    default=False,
    action="store_true",
    help="use chat template or not. Default to False.",
  )
  parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="an addtidional tag in save name",
  )
  parser.add_argument(
    "--attention",
    type=str,
    default="flash",
    help="attention type, default to flash",
  )
  parser.add_argument(
    "--cfg",
    type=str,
    default="",
    help="addtional configs for model, for example 'cfg_name_1=cfg_value_1,cfg_name_2=cfg_value_2', default to ''",
  )
  parser.add_argument(
    "--p",
    type=float,
    default=0.9,
    help=" ",
  )

  args = parser.parse_args()
  return args


def str_to_dict(cfg: str):
  cfg = [c.strip() for c in cfg.split(",") if c.strip() != ""]
  cfg = {c.split("=")[0]: c.split("=")[1] if "=" in c else True for c in cfg}
  for k, v in cfg.items():
    try:
      cfg[k] = literal_eval(v)
    except:
      cfg[k] = v
  return cfg


def tok_encode_middle_trunc(
  self: HFLM, string: str, left_truncate_len=None, add_special_tokens=None
) -> List[int]:
  """ """
  special_tokens_kwargs = {}

  if add_special_tokens is None:
    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
      special_tokens_kwargs = {"add_special_tokens": False or self.add_bos_token}
  else:
    special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

  encoding = self.tokenizer.encode(string, **special_tokens_kwargs)


  if len(encoding) > self.max_length:
    half = int(self.max_length / 2)
    encoding = encoding[:half] + encoding[-half:]

  return encoding


def mid_trunc_and_pad(x: list, max_len: int, pad_len: int, pad_value: int):
  half_len = max_len // 2
  if len(x) > max_len:
    x = x[:half_len] + x[-half_len:]
  x = torch.tensor(x)
  if len(x) < pad_len:
    x = F.pad(x, (pad_len - x.shape[0], 0), value=pad_value)
  return x


def tok_batch_encode_middle_trunc(
  self: HFLM,
  strings: List[str],
  padding_side: str = "left",
  left_truncate_len: int = None,
  truncation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
  old_padding_side = self.tokenizer.padding_side
  self.tokenizer.padding_side = padding_side

  add_special_tokens = {}
  if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
    add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

  encoding = self.tokenizer(
    strings,
    **add_special_tokens,
  )

  pad_len = min(max(len(x) for x in encoding["input_ids"]), self.max_length)
  encoding["input_ids"] = torch.stack(
    list(
      map(
        partial(
          mid_trunc_and_pad,
          max_len=self.max_length,
          pad_len=pad_len,
          pad_value=self.tokenizer.pad_token_id,
        ),
        encoding["input_ids"],
      )
    ),
    dim=0,
  )
  encoding["attention_mask"] = torch.stack(
    list(
      map(
        partial(
          mid_trunc_and_pad,
          max_len=self.max_length,
          pad_len=pad_len,
          pad_value=0,
        ),
        encoding["attention_mask"],
      )
    ),
    dim=0,
  )

  self.tokenizer.padding_side = old_padding_side
  return encoding["input_ids"], encoding["attention_mask"]


def convert_to_json_compatible(data):
  if isinstance(data, dict):
    return {key: convert_to_json_compatible(value) for key, value in data.items()}
  elif isinstance(data, list):
    return [convert_to_json_compatible(item) for item in data]
  elif isinstance(data, (str, int, float, bool)) or data is None:
    return data
  else:
    return str(data)


def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.cuda.manual_seed_all(seed)


def fixed_generate_until(
  self, requests: List[Instance], disable_tqdm: bool = False
) -> List[str]:
  res = []

  def _collate(req: Tuple[str, dict]):
    """Defines the key for the sorted method"""
    toks = self.tok_encode(req[0])
    return -len(toks), req[0]

  pbar = tqdm(
    total=len(requests),
    disable=(disable_tqdm or (self.rank != 0)),
    desc="Running generate_until requests",
  )
  adaptive_batch_size = None
  if self.batch_size == "auto":
    print("Passed argument batch_size = auto. Detecting largest batch size")
    batch_size = self._detect_batch_size()
    print(f"Determined Largest batch size: {batch_size}")
    adaptive_batch_size = batch_size
  batch_size = (
    self.batch_size
    if self.batch_size != "auto"
    else adaptive_batch_size if adaptive_batch_size is not None else 0
  )
  batch_fn = (
    self._batch_scheduler
    if self.batch_size == "auto" and not adaptive_batch_size
    else None
  )

  re_ords = Collator(
    [reg.args for reg in requests],
    sort_fn=_collate,
    group_by="gen_kwargs",
    group_fn=lambda x: x[1],
  )
  chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
  for chunk in chunks:
    contexts, all_gen_kwargs = zip(*chunk)
    gen_kwargs = all_gen_kwargs[0]
    until = None
    if isinstance(gen_kwargs, dict):
      kwargs = copy.deepcopy(gen_kwargs) # edge case for repeats > 1
      if "until" in kwargs.keys():
        until = kwargs.pop("until")
        if isinstance(until, str):
          until = [until]
        elif not isinstance(until, list):
          raise ValueError(
            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
          )
    else:
      raise ValueError(
        f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
      )
    eos = self.tok_decode([self.eot_token_id], skip_special_tokens=False)
    if not until:
      until = [eos]
    else:
      until.append(eos)
    if "max_gen_toks" in kwargs.keys():
      max_gen_toks = kwargs.pop("max_gen_toks")
    else:
      max_gen_toks = self.max_gen_toks

    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
      max_ctx_len = self.max_length - max_gen_toks
    elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
      max_ctx_len = self.max_length

    context_enc, attn_masks = self.tok_batch_encode(
      contexts,
      left_truncate_len=max_ctx_len,
      truncation=self.truncation,
    )
    context_enc = context_enc.to(self.device)
    attn_masks = attn_masks.to(self.device)

    if "max_length" not in kwargs:
      kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

    cont = self._model_generate(
      context=context_enc,
      attention_mask=attn_masks,
      stop=until,
      **kwargs,
    )

    cont_toks_list = cont.tolist()
    for cont_toks, context in zip(cont_toks_list, contexts):
      if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
        cont_toks = cont_toks[context_enc.shape[1] :]

      s = self.tok_decode(cont_toks)

      for term in until:
        if len(term) > 0:
          s = s.split(term)[0]

      res.append(s)

      self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
      pbar.update(1)
  res = re_ords.get_original(res)

  pbar.close()

  return res


TIMER_SECOND = {}
TIMER_COUNT = {}
TIME_STAMP = 0


def reset_timer():
  global TIMER_SECOND
  global TIMER_COUNT
  TIMER_SECOND = {}
  TIMER_COUNT = {}


def get_avg_time(name=None):
  global TIMER_SECOND
  global TIMER_COUNT
  if name in TIMER_COUNT and TIMER_COUNT[name] > 0:
    return TIMER_SECOND[name] / TIMER_COUNT[name]
  else:
    return None


def tik():
  global TIME_STAMP
  torch.cuda.synchronize()
  TIME_STAMP = time()


def tok(name=None):
  global TIME_STAMP
  torch.cuda.synchronize()
  update_timer(name, time() - TIME_STAMP)


def update_timer(name, value):
  global TIMER_SECOND
  global TIMER_COUNT
  if name not in TIMER_SECOND:
    TIMER_SECOND[name] = 0
    TIMER_COUNT[name] = 0
  TIMER_SECOND[name] += value
  TIMER_COUNT[name] += 1
