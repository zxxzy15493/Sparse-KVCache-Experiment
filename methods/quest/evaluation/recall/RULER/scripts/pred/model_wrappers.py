#
#
#

import json
import logging
import requests
import torch
from typing import Dict, List, Optional
try:
  from transformers.models.llama.modeling_llama import LlamaForCausalLM as _HF_LlamaForCausalLM

  _old_llama_forward = _HF_LlamaForCausalLM.forward

  def _patched_llama_forward(self, *args, **kwargs):

    kwargs.pop("cache_position", None)
    return _old_llama_forward(self, *args, **kwargs)

  _HF_LlamaForCausalLM.forward = _patched_llama_forward
except Exception as e:

  print("[WARN] Failed to patch LlamaForCausalLM.forward:", e)

class HuggingFaceModel:
  def __init__(self, name_or_path: str, **generation_kwargs) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

    if 'Yarn-Llama' in name_or_path:
      model_kwargs = None
    else:
      model_kwargs = {"attn_implementation": "flash_attention_2"}
    
    #   )
    
    self.pipeline = None
    self.model = AutoModelForCausalLM.from_pretrained(
      name_or_path, 
      trust_remote_code=True, 
      device_map="auto", 
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2"
      )
    

    old_forward = self.model.forward

    def patched_forward(*args, **kwargs):

      kwargs.pop("cache_position", None)
      return old_forward(*args, **kwargs)

    self.model.forward = patched_forward
    print("[INFO] Patched model.forward to ignore cache_position")

    
    self.model.config.pad_token_id = self.tokenizer.eos_token_id
    self.generation_kwargs = generation_kwargs 
    self.stop = self.generation_kwargs.pop('stop')

    if self.tokenizer.pad_token is None:
      self.tokenizer.padding_side = 'left'
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


  def __call__(self, prompt: str, **kwargs) -> dict:
    return self.process_batch([prompt], **kwargs)[0]

  def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
    if self.pipeline is None:
      inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
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
