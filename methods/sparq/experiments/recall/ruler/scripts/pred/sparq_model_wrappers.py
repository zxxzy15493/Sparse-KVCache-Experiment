#
#
#

import json
import logging
import requests
import torch
from typing import Dict, List, Optional,Any

from llminference.myexperiments import Sparsity,SparsityMethods
from llminference.methods.ann_attention_copy import set_current_sample_id

class HuggingFaceModel:
  def __init__(self, name_or_path: str, sparsity:Optional[Dict[str, Any]] = None, **generation_kwargs) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    
    self.pipeline = None
    self.model = AutoModelForCausalLM.from_pretrained(
      name_or_path, 
      trust_remote_code=True, 
      device_map="auto", 
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
    )
    self.model.config.pad_token_id = self.tokenizer.eos_token_id

    self.model._debug_tokenizer = self.tokenizer

    sparq = Sparsity(
      name=sparsity["name"],
      k=sparsity["k"],
      local_k=sparsity["local_k"],
      rank=sparsity["rank"],
      score=sparsity["score"],
      reallocate_to_mean_value=sparsity["reallocate_to_mean_value"],
      type=sparsity.get("type", "recall"),
      recall_save_path=sparsity.get("recall_save_path"),
    )
    self.model = SparsityMethods.apply(sparq, self.model)


    self.generation_kwargs = generation_kwargs
    self.stop = self.generation_kwargs.pop('stop')

    if self.tokenizer.pad_token is None:
      self.tokenizer.padding_side = 'left'
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


  def __call__(self, prompt: str, **kwargs) -> dict:
    return self.process_batch([prompt], **kwargs)[0]
  
  def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
    sample_ids = kwargs.pop("sample_ids", None)
    if sample_ids is not None and len(prompts) > 1:
      results = []
      for prompt, sample_id in zip(prompts, sample_ids):
        results.extend(self.process_batch([prompt], sample_ids=[sample_id], **kwargs))
      return results
    if sample_ids is not None:
      set_current_sample_id(self.model, sample_ids[0])

    if self.pipeline is None:
      inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
      print("tokenizer.model_max_length =", self.tokenizer.model_max_length)
      print("model max_position_embeddings =", getattr(self.model.config, "max_position_embeddings", None))

      generated_ids = self.model.generate(
        **inputs,
        **self.generation_kwargs
      )
      generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    else:
      output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
      assert len(output) == len(prompts)
      generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

    results = []

    for text, prompt in zip(generated_texts, prompts):
      if self.pipeline is None:    
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
        prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
      
      if text.startswith(prompt):
        text = text[len(prompt):]

      if self.stop is not None:
        for s in self.stop:
          text = text.split(s)[0]

      results.append({'text': [text]})

    return results

class MambaModel:
  def __init__(self, name_or_path: str, **generation_kwargs) -> None:
    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    self.device = "cuda"
    self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
    self.generation_kwargs = generation_kwargs
    self.stop = self.generation_kwargs.pop('stop')
    self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
    self.minp = 0.0

  def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
    tokens = self.tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(self.device)
    max_length = input_ids.shape[1] + self.max_genlen

    out = self.model.generate(
      input_ids=input_ids,
      max_length=max_length,
      cg=True,
      return_dict_in_generate=True,
      output_scores=True,
      enable_timing=False,
      **self.generation_kwargs,
    )
    assert len(out.sequences) == 1
    return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

  def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
    return [self.__call__(prompt, **kwargs) for prompt in prompts]
