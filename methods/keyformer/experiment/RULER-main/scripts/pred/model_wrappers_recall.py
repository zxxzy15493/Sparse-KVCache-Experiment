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
        self.recent_size = generation_kwargs.pop('recent_size', 32)

        self.enable_keyformer = generation_kwargs.pop('enable_keyformer', False)
        self.key_size = generation_kwargs.pop('key_size', 992)
        self.tau_init = generation_kwargs.pop('tau_init', 1.0)
        self.tau_delta = generation_kwargs.pop('tau_delta', 0.01)

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        if self.enable_keyformer:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                **({} if model_kwargs is None else model_kwargs)
            )

            class SubArgs:
                pass

            from keyformer_kv.keyformer_recall import enable_keyformer_recall
            kf_args = SubArgs()
            kf_args.key_size = self.key_size
            kf_args.recent_size = self.recent_size
            kf_args.tau_init = self.tau_init
            kf_args.tau_delta = self.tau_delta
            kf_args.check_recall = self.check_recall

            enable_keyformer_recall(self.model, kf_args)
            print(f"RULER KeyformerKVInjected. KeySize:{self.key_size}, Recent:{self.recent_size}")

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
        if self.check_recall and self.enable_keyformer:
            from keyformer_kv.keyformer_recall import init_new_sample_registry
            init_new_sample_registry()

        for module in self.model.modules():
            if hasattr(module, "kv_cache") and module.kv_cache is not None:
                module.kv_cache.select_idx = None
                module.kv_cache._global_step = 0
                module.kv_cache.absolute_indices = None
                module.kv_cache.shadow_key = None
                module.kv_cache.total_processed_tokens = 0

                if hasattr(module.kv_cache, "key_score"):
                    module.kv_cache.key_score = None

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        results = []

        for prompt in prompts:
            if self.enable_keyformer:
                self.reset_sample_cache_state()

            if self.pipeline is None:
                inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)
                context_length = inputs.input_ids.shape[1]

                gen_kwargs = self.generation_kwargs.copy()
                gen_kwargs['num_beams'] = 1
                gen_kwargs['min_length'] = context_length + 1
                gen_kwargs['num_logits_to_keep'] = 1

                if not gen_kwargs.get('do_sample', False) or gen_kwargs.get('temperature', 1.0) == 0.0:
                    gen_kwargs['do_sample'] = False
                    gen_kwargs.pop('temperature', None)
                    gen_kwargs.pop('top_k', None)
                    gen_kwargs.pop('top_p', None)

                generated_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs
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
