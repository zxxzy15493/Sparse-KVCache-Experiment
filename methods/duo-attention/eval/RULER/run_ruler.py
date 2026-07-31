#!/usr/bin/env python3

import os
import sys
import json
import random
import argparse
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
import torch
import traceback
from typing import Optional, List, Union, Tuple, Dict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig
)


import transformers


try:
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    original_llama_forward = LlamaForCausalLM.forward
    
    def patched_llama_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs, 
    ):

        kwargs.pop('cache_position', None)
        return original_llama_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    

    LlamaForCausalLM.forward = patched_llama_forward
    print("已应用LlamaForCausalLM forward兼容性补丁")
except Exception as e:
    print(f"应用Llama补丁时出错: {e}")


try:
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    original_mistral_forward = MistralForCausalLM.forward
    
    def patched_mistral_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        kwargs.pop('cache_position', None)
        return original_mistral_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    MistralForCausalLM.forward = patched_mistral_forward
    print("已应用MistralForCausalLM forward兼容性补丁")
except Exception as e:
    print(f"应用Mistral补丁时出错: {e}")


try:
    from duo_attn.patch import enable_duo_attention_eval
    from duo_attn.utils import (
        load_attn_pattern,
        sparsify_attention_heads,
    )
    from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
    DUO_ATTN_AVAILABLE = True
    print("DuoAttention模块已成功导入")
except ImportError as e:
    print(f"Warning: duo_attn module not found: {e}")
    print("Make sure duo_attn is installed or in PYTHONPATH")
    DUO_ATTN_AVAILABLE = False


RULER_TASKS = {
    'niah': {
        'tokens_to_generate': 128,
        'template': """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        'answer_prefix': """ The special magic {type_needle_v} for {query} mentioned in the provided text are"""
    },
    'variable_tracking': {
        'tokens_to_generate': 30,
        'template': """Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.""",
        'answer_prefix': """ Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: """
    },
    'common_words_extraction': {
        'tokens_to_generate': 120,
        'template': """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?""",
        'answer_prefix': """ Answer: The top 10 words that appear most often in the list are:"""
    },
    'freq_words_extraction': {
        'tokens_to_generate': 50,
        'template': """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
        'answer_prefix': """ Answer: According to the coded text above, the three most frequently appeared words are:"""
    },
    'qa': {
        'tokens_to_generate': 32, 
        'template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
        'answer_prefix': """ Answer:""",
    },
}


def read_manifest(file_path):
    """读取JSONL格式的数据文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {file_path}: {line[:100]}...")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="RULER数据集评估脚本 - 兼容性修复版")

    parser.add_argument("--task_data", type=str, required=True, help="任务数据文件路径")
    parser.add_argument("--save_dir", type=str, required=True, help="保存预测结果的目录")
    parser.add_argument("--benchmark", type=str, default='synthetic', help="基准名称")
    parser.add_argument("--task", type=str, required=True, help="任务名称")
    

    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="使用快速分词器")
    

    parser.add_argument("--method", type=str, default="full", choices=["full", "duo_attn"])
    parser.add_argument("--attn_load_dir", type=str, default=None, help="注意力模式目录路径")
    parser.add_argument("--sink_size", type=int, default=None, help="sink token数量")
    parser.add_argument("--recent_size", type=int, default=None, help="recent token数量")
    parser.add_argument("--sparsity", type=float, default=0.5, help="注意力稀疏度")
    parser.add_argument("--max_capacity_prompts", type=int, default=1024, help="最大有效容量")
    

    parser.add_argument("--max_new_tokens", type=int, default=None, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成温度")
    parser.add_argument("--do_sample", action="store_true", help="是否使用采样生成")
    

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_examples", type=int, default=-1, help="最大评估样本数")
    parser.add_argument("--resume", action="store_true", help="从已有预测文件恢复")
    parser.add_argument("--use_cache", action="store_true", default=True, help="使用KV缓存")
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args):
    print(f"加载模型: {args.model_name} from {args.model_path}")
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )
    

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation="eager"  
    )
    

    def wrap_model_forward(model):
        original_forward = model.forward
        
        def new_forward(*args, **kwargs):
   
            kwargs.pop('cache_position', None)
            kwargs.pop('position_ids', None)
            return original_forward(*args, **kwargs)
        
        model.forward = new_forward

        for name, module in model.named_modules():
            if name == '':  
                continue
            original_module_forward = module.forward
            
            def make_new_module_forward(original):
                def new_module_forward(*args, **kwargs):
                    kwargs.pop('cache_position', None)
                    kwargs.pop('position_ids', None)
                    return original(*args, **kwargs)
                return new_module_forward
            
            module.forward = make_new_module_forward(original_module_forward)
        
        return model
    
    model = wrap_model_forward(model)
    

    generation_config = GenerationConfig.from_pretrained(args.model_path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]
    
    model.eval()
    
   
    if args.method == "duo_attn" and DUO_ATTN_AVAILABLE:
        assert args.attn_load_dir is not None, "attn_load_dir must be provided for duo_attn method"
        print(f"从 {args.attn_load_dir} 加载注意力模式，稀疏度 {args.sparsity}")
        
        full_attention_heads, sink_size, recent_size = load_attn_pattern(args.attn_load_dir)

        try:
            model._orig_full_attention_heads = full_attention_heads.copy()
        except Exception:
            model._orig_full_attention_heads = full_attention_heads


        max_capacity = args.max_capacity_prompts
        sink_size = args.sink_size if args.sink_size is not None else 128
        recent_size = args.recent_size if args.recent_size is not None else 256


        fixed_sparsity = args.sparsity
        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, None, sparsity=fixed_sparsity
        )
        print(f"实际稀疏度: {sparsity} (固定为 {fixed_sparsity})")
        
        enable_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
        

        try:
            total_heads = len(full_attention_heads)
            kept_heads = sum(1 for h in full_attention_heads if bool(h))
        except Exception:
            total_heads = 'NA'
            kept_heads = 'NA'

        print(
            f"#sym:duo_enabled sink={sink_size} recent={recent_size} "
            f"max_capacity_prompts={max_capacity} fixed_sparsity={fixed_sparsity} "
            f"total_heads={total_heads} kept_heads_before_apply={kept_heads} applied_sparsity={sparsity}"
        )

        print("DuoAttention已成功启用")
    elif args.method == "full" and DUO_ATTN_AVAILABLE:
        enable_tuple_kv_cache(model)
        print("启用tuple KV缓存（全注意力模式）")
    
    return model, tokenizer, eos_token_ids


def load_ruler_config(task, benchmark, config_dir=None):
    """加载RULER任务配置"""
    task_config = RULER_TASKS.get(task, {}).copy()
    
    if config_dir and os.path.exists(os.path.join(config_dir, f"{benchmark}.yaml")):
        yaml_path = os.path.join(config_dir, f"{benchmark}.yaml")
        try:
            with open(yaml_path, "r") as f:
                tasks_customized = yaml.safe_load(f)
            
            if task in tasks_customized:
                task_config.update(tasks_customized[task])
        except Exception as e:
            print(f"Warning: Failed to load yaml config from {yaml_path}: {e}")
    
    return task_config


def format_prompt(data_point, task_config):
    """格式化提示"""
    if 'input' in data_point and data_point['input']:
        return data_point['input']
    elif 'context' in data_point and 'query' in data_point:
        template = task_config.get('template', '{context}\n\nQuestion: {query}\nAnswer:')
        try:
            return template.format(**data_point)
        except KeyError as e:
            print(f"Warning: Missing key in template: {e}")
            return f"{data_point.get('context', '')}\n\nQuestion: {data_point.get('query', '')}\nAnswer:"
    else:
        return str(data_point)


def get_prediction(model, tokenizer, prompt, task_config, args):
    """获取单个样本的预测 - 简化版本避免cache_position问题"""

    tokenized_prompt = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=False,
        add_special_tokens=True
    ).to(model.device)
    
    context_length = tokenized_prompt.input_ids.shape[-1]
    

    max_new_tokens = args.max_new_tokens or task_config.get('tokens_to_generate', 100)

    with torch.no_grad():

        input_ids = tokenized_prompt.input_ids
        attention_mask = tokenized_prompt.attention_mask
        

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        
        if args.do_sample and args.temperature > 0:

            probs = torch.softmax(next_token_logits / args.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        generated_tokens = [next_token.item()]
        

        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=next_token,
                attention_mask=torch.ones_like(next_token),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            if args.do_sample and args.temperature > 0:
                probs = torch.softmax(next_token_logits / args.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token.item())
            

            if next_token.item() == tokenizer.eos_token_id:
                break
        
 
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_len = context_length + len(generated_tokens)

    return generated_text, context_length, output_len, {'overall_recall': 0.0, 'layerwise_recall': {}}


def evaluate_ruler(args):
    """在RULER数据集上进行评估"""
    set_seed(args.seed)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    

    config_dir = os.path.dirname(args.task_data) if args.task_data else None
    task_config = load_ruler_config(args.task, args.benchmark, config_dir)
    

    print(f"加载数据: {args.task_data}")
    if not os.path.exists(args.task_data):
        raise FileNotFoundError(f"Data file not found: {args.task_data}")
    
    data = read_manifest(args.task_data)
    print(f"加载了 {len(data)} 个样本")
    
    if args.max_examples > 0 and args.max_examples < len(data):
        data = data[:args.max_examples]
        print(f"限制到 {len(data)} 个样本")
    

    model, tokenizer, eos_token_ids = load_model_and_tokenizer(args)
    

    pred_file = save_dir / f'{args.task}.jsonl'
    

    if args.resume and os.path.exists(pred_file):
        print(f"恢复模式: 已存在预测文件 {pred_file}")
        predicted_indices = set()
        try:
            existing_data = read_manifest(pred_file)
            predicted_indices = {item.get('index', -1) for item in existing_data}
            print(f"已预测 {len(predicted_indices)} 个样本")
        except Exception as e:
            print(f"Warning: Failed to read existing predictions: {e}")
        
        data = [item for item in data if item.get('index', -1) not in predicted_indices]
        print(f"剩余 {len(data)} 个样本需要预测")
    else:
        print(f"新运行: 将预测所有 {len(data)} 个样本")
    
    if not data:
        print("没有需要预测的样本")
        return
    

    total_tokens = 0
    predictions = []
    flushed_count = 0
    
    print(f"开始评估任务 {args.task}...")
    pbar = tqdm(data, desc=f"预测 {args.task}")
    
    for data_point in pbar:
        
        try:

            prompt = format_prompt(data_point, task_config)
            

            generated_text, input_len, output_len, _ = get_prediction(
                model, tokenizer, prompt, task_config, args
            )

            result = {
                'index': data_point.get('index', -1),
                'pred': generated_text,
                'input': prompt,
                'outputs': data_point.get('outputs', []),
                'others': data_point.get('others', {}),
                'truncation': data_point.get('truncation', -1),
                'length': data_point.get('length', -1),
            }
            
            predictions.append(result)
            total_tokens += output_len
            

            if len(predictions) % 10 == 0:
                with open(pred_file, 'a', encoding='utf-8', buffering=1) as fout:
                    for pred in predictions[flushed_count:]:
                        fout.write(json.dumps(pred) + '\n')
                flushed_count = len(predictions)

            if len(predictions) % 10 == 0:
                print(f"已处理 {len(predictions)}/{len(data)} 个样本")
        
        except Exception as e:
            print(f"处理样本 {data_point.get('index', 'unknown')} 时出错: {e}")
            traceback.print_exc()
            continue
    

    if flushed_count < len(predictions):
        with open(pred_file, 'a', encoding='utf-8', buffering=1) as fout:
            for pred in predictions[flushed_count:]:
                fout.write(json.dumps(pred) + '\n')

    total_samples = len(predictions)
    if total_samples > 0:
        avg_tokens_per_sample = total_tokens / total_samples
    else:
        avg_tokens_per_sample = 0
    
    print("\n" + "="*80)
    print(f"评估完成!")
    print(f"任务: {args.task}")
    print(f"总样本数: {total_samples}")
    print(f"总token数: {total_tokens}")
    print(f"平均每样本token数: {avg_tokens_per_sample:.1f}")
    print(f"预测结果保存到: {pred_file}")
   
def main():
    args = parse_args()
    
    if args.method == "duo_attn" and not DUO_ATTN_AVAILABLE:
        print("错误: duo_attn模块不可用")
        print("请确保已正确安装duo_attn或将其添加到PYTHONPATH")
        sys.exit(1)
    
    print("="*80)
    print("RULER数据集评估 - 兼容性修复版")
    print("="*80)
    print(f"任务: {args.task}")
    print(f"模型: {args.model_name} ({args.model_path})")
    print(f"方法: {args.method}")
    if args.method == "duo_attn":
        print(f"注意力目录: {args.attn_load_dir}")
        print(f"稀疏度: {args.sparsity}")
    print(f"数据文件: {args.task_data}")
    print(f"保存目录: {args.save_dir}")
    print("="*80)
    
    evaluate_ruler(args)

if __name__ == "__main__":
    main()