import json
import logging
import requests
import torch
import os
from typing import Dict, List, Optional

class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        self.name_or_path = name_or_path
        self.check_recall = generation_kwargs.pop('check_recall', False)
        self.enable_h2o = generation_kwargs.pop('enable_h2o', False)
        self.recent_size = generation_kwargs.pop('recent_size', 32)
        self.heavy_hitter_size = generation_kwargs.pop('heavy_hitter_size', 992)

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        if self.enable_h2o:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                **({} if model_kwargs is None else model_kwargs)
            )

            from h2o_kv.h2o_recall import enable_h2o_recall
            class SubArgs:
                pass
            h2o_args = SubArgs()
            h2o_args.heavy_hitter_size = self.heavy_hitter_size
            h2o_args.recent_size = self.recent_size
            h2o_args.check_recall = self.check_recall

            enable_h2o_recall(self.model, h2o_args)
            print(f"RULER H2OKVCache injected successfully. Layers patched.")

        else:
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=name_or_path,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=model_kwargs,
                )
                self.model = self.pipeline.model
            except Exception:
                self.pipeline = None
                self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop', None)

        if self.tokenizer.pad_token is None:
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def reset_sample_cache_state(self):
        if self.check_recall:
            from h2o_kv.h2o_recall import init_new_sample_registry
            init_new_sample_registry()

        for module in self.model.modules():
            if hasattr(module, "kv_cache") and module.kv_cache is not None:
                module.kv_cache.hh_score = None
                module.kv_cache.select_idx = None
                module.kv_cache._global_step = 0

                module.kv_cache.absolute_indices = None
                module.kv_cache.shadow_key = None
                module.kv_cache.total_processed_tokens = 0

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        results = []

        for prompt in prompts:
            if self.enable_h2o:
                self.reset_sample_cache_state()
            self.generation_kwargs['num_logits_to_keep'] = 1
            if self.pipeline is None:
                inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generation_kwargs
                )
                text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                output = self.pipeline(text_inputs=[prompt], **self.generation_kwargs)
                text = output[0][0]["generated_text"]

            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt_text = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            else:
                prompt_text = prompt

            if text.startswith(prompt_text):
                text = text[len(prompt_text):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results
