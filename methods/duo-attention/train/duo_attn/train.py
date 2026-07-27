import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import wandb
import matplotlib.pyplot as plt
from duo_attn.utils import (
    get_model,
    parse_args,
    get_tokenizer,
    visualize_pruned_attention_heads,
    full_attention_heads_to_list,
    save_full_attention_heads,
    seed_everything,
)
from duo_attn.data import (
    get_dataset,
    MultiplePasskeyRetrievalDataset,
    get_supervised_dataloader,
)
from duo_attn.patch import (
    enable_duo_attention_training,
    get_full_attention_heads,
    set_full_attention_heads,
    map_full_attention_heads,
    load_full_attention_heads,
)

from duo_attn.loss import l1_loss


import torch.distributed as dist

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
import types

from transformers import AutoModelForCausalLM, AutoConfig

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)




def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def apply_fsdp(model: torch.nn.Module, mesh, mp_policy, modules_to_shard):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}

    for module in model.modules():
        if any([isinstance(module, m) for m in modules_to_shard]):
            fully_shard(module, **fsdp_config)
    fully_shard(model, **fsdp_config)


def get_model_backbone(model: torch.nn.Module):
    if hasattr(model, "model") and model.model is not None:
        return model.model
    if hasattr(model, "transformer") and model.transformer is not None:
        return model.transformer
    return model


def get_modules_to_shard(model_backbone):
    def _unwrap_layers_container(container):
        current = container
        for _ in range(4):
            if current is None:
                return None
            if hasattr(current, "module"):
                current = current.module
                continue
            if hasattr(current, "wrapped_module"):
                current = current.wrapped_module
                continue
            break
        return current

    def _first_layer_type(container):
        container = _unwrap_layers_container(container)
        if container is None:
            return None
        try:
            if len(container) > 0:
                return type(container[0])
        except TypeError:
            return None
        except Exception:
            return None
        return None

    modules_to_shard = {LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer}
    layers = getattr(model_backbone, "layers", None)
    layer_type = _first_layer_type(layers)
    if layer_type is not None:
        modules_to_shard.add(layer_type)

    encoder = getattr(model_backbone, "encoder", None)
    encoder_layers = getattr(encoder, "layers", None) if encoder is not None else None
    encoder_layer_type = _first_layer_type(encoder_layers)
    if encoder_layer_type is not None:
        modules_to_shard.add(encoder_layer_type)

    return modules_to_shard


def train(
    args, model, rank, world_size, train_dataloader, optimizer, scheduler, resume_step
):
    model.train()

    if rank == 0:
        pbar = tqdm(range(args.num_steps))

    local_rank = int(os.environ["LOCAL_RANK"])

    global_step = 0
    local_step = 0

    def _maybe_all_reduce_scalar(tensor):
        if world_size <= 1:
            return tensor
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        return tensor

    while True:
        if global_step >= args.num_steps:
            break
        for step, batch in enumerate(train_dataloader):
            if global_step <= resume_step:
                global_step += 1
                if rank == 0:
                    pbar.update(1)
                    pbar.set_description(
                        f"Skipping step {global_step} to resume to {resume_step}"
                    )
                continue

            @torch.no_grad()
            def clamp_(x, min_val, max_val):
                x.clamp_(min_val, max_val)

            map_full_attention_heads(model, func=lambda x: clamp_(x, 0, 1))

            batch = {k: v.to(f"cuda:{local_rank}") for k, v in batch.items()}

            # duplicate for the two way forward
            input_ids = torch.cat([batch["input_ids"], batch["input_ids"]], dim=0)

            seq_len = input_ids.shape[1]
            seq_parallel_chunk_size = seq_len // world_size
            seq_parallel_chunk_start = seq_parallel_chunk_size * rank
            seq_parallel_chunk_end = seq_parallel_chunk_start + seq_parallel_chunk_size
            position_ids = torch.arange(
                seq_parallel_chunk_start,
                seq_parallel_chunk_end,
                device=input_ids.device,
            ).unsqueeze(0)

            outputs = model(
                input_ids=input_ids[:, seq_parallel_chunk_start:seq_parallel_chunk_end],
                position_ids=position_ids,
                use_cache=False,
            )

            hidden_states = outputs[0]

            original_hidden_states = hidden_states[: args.batch_size]
            pruned_hidden_states = hidden_states[args.batch_size :]

            labels = batch["labels"][:, seq_parallel_chunk_start:seq_parallel_chunk_end]
            label_mask = labels != -100
            num_labels = label_mask.sum()
            global_num_labels = num_labels.clone().detach()
            dist.all_reduce(global_num_labels)

            # debug print for label counts
            if os.environ.get("DUO_DEBUG", "0") != "0" and rank == 0:
                try:
                    print("[DEBUG] num_labels (local):", int(num_labels.item()), "global_num_labels:", int(global_num_labels.item()))
                    print("[DEBUG] label_mask sum (local):", int(label_mask.sum().item()))
                except Exception:
                    pass

            # filter out label == IGNORE_INDEX (-100)
            original_hidden_states = original_hidden_states[label_mask].float()
            pruned_hidden_states = pruned_hidden_states[label_mask].float()

            distill_loss = (
                (original_hidden_states - pruned_hidden_states)
                .pow(2)
                .mean(dim=-1)
                .sum()
                * world_size
                / global_num_labels
            )

            def _to_local_tensor(tensor):
                if hasattr(tensor, "to_local"):
                    return tensor.to_local()
                return tensor

            # Obtain the actual parameter references for full_attention_heads so gradients flow
            full_attention_params = [
                param
                for name, param in model.named_parameters()
                if "full_attention_heads" in name and param.requires_grad
            ]

            # For visualization / saving, use detached tensors on the appropriate device
            full_attention_heads = [
                _to_local_tensor(p).detach().to(original_hidden_states.device)
                for p in full_attention_params
            ]

            # Regularize toward 1.0 for full-attention heads (encourage enabling full attention)
            # Use the parameter tensors directly (not detached) so gradients propagate
            if len(full_attention_params) > 0:
                reg_targets = torch.cat(
                    [_to_local_tensor(p).reshape(-1) for p in full_attention_params]
                ).float().to(original_hidden_states.device)
                reg_loss = l1_loss(reg_targets)
            else:
                reg_loss = torch.tensor(0.0, device=original_hidden_states.device)

            if args.reg_weight == 0:
                loss = distill_loss
            else:
                loss = distill_loss + args.reg_weight * reg_loss

            # optional debug prints controlled by env DUO_DEBUG
            debug = os.environ.get("DUO_DEBUG", "0") != "0"
            if debug and rank == 0:
                try:
                    print("[DEBUG] original_hidden_states mean,std:", original_hidden_states.mean().item(), original_hidden_states.std().item())
                    print("[DEBUG] pruned_hidden_states mean,std:", pruned_hidden_states.mean().item(), pruned_hidden_states.std().item())
                    print("[DEBUG] diff mean,abs_mean:", (original_hidden_states - pruned_hidden_states).mean().item(), (original_hidden_states - pruned_hidden_states).abs().mean().item())
                    # full_attention_heads may be tensors or custom objects
                    sample_heads = []
                    for h in full_attention_heads:
                        try:
                            arr = h.detach().float().cpu().numpy().ravel()[:8]
                        except Exception:
                            arr = None
                        sample_heads.append(arr)
                    print("[DEBUG] sample full_attention_heads:", sample_heads[:3])
                except Exception as e:
                    print("[DEBUG] failed to print debug info:", e)

            loss.backward()

            local_step = (local_step + 1) % args.gradient_accumulation_steps

            _maybe_all_reduce_scalar(loss)
            _maybe_all_reduce_scalar(distill_loss)
            _maybe_all_reduce_scalar(reg_loss)

            if local_step != 0:
                continue

            if debug and rank == 0:
                # print gradient norm for first full_attention_heads param to confirm gradient flow
                try:
                    for name, param in model.named_parameters():
                        if "full_attention_heads" in name:
                            g = param.grad
                            print(f"[DEBUG] grad for {name}:", None if g is None else float(g.norm()))
                            break
                except Exception as e:
                    print("[DEBUG] failed to print grad info:", e)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if rank == 0:
                full_attention_heads_list = full_attention_heads_to_list(
                    full_attention_heads
                )

                if not args.disable_wandb:
                    fig = visualize_pruned_attention_heads(full_attention_heads_list)

                    sample_len = batch["input_ids"].shape[1]
                    wandb.log(
                        {
                            "distill_loss": distill_loss.item(),
                            "reg_loss": reg_loss.item(),
                            "attn_heads": fig,
                            "step": global_step,
                            "sample_len": sample_len,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                    plt.close(fig)

                pbar.set_description(
                    f"Len={seq_len}/{global_num_labels}|Dloss={distill_loss.item():.3f}|Rloss={reg_loss.item():.3f}|LR={optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.update(1)

            if args.output_dir is not None and global_step % args.save_steps == 0:
                if rank == 0:
                    save_full_attention_heads(
                        full_attention_heads_list,
                        os.path.join(
                            args.output_dir,
                            f"full_attention_heads_step={global_step}.tsv",
                        ),
                    )
                    os.system(f"rm {args.output_dir}/full_attention_heads_latest.tsv")
                    os.system(
                        f"cp {args.output_dir}/full_attention_heads_step={global_step}.tsv {args.output_dir}/full_attention_heads_latest.tsv"
                    )

                # save scheduler and optimizer state
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(
                        args.output_dir,
                        f"optimizer_scheduler_state-step={global_step}-rank={rank}.pt",
                    ),
                )

                # copy the full_attention_heads and optimizer_scheduler_state to the latest state, replacing the old one
                # remove the previous latest state
                os.system(
                    f"rm {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )
                os.system(
                    f"cp {args.output_dir}/optimizer_scheduler_state-step={global_step}-rank={rank}.pt {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )

            if global_step >= args.num_steps:
                break

    if rank == 0:
        pbar.close()


def main(args):
    print(f"[DEBUG] Starting execution; model: {args.model_name}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    print(f"[DEBUG] GPU count: {torch.cuda.device_count()}")
    print(f"[DEBUG] Process environment — RANK: {os.environ.get('RANK')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.model_name)

    if args.config_name is not None:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    if args.rope_theta is not None:
        print(f"Setting rope_theta from {config.rope_theta} to {args.rope_theta}")
        config.rope_theta = args.rope_theta


    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    def is_glm_model(model_path):
        return "glm" in model_path.lower()

    def maybe_patch_chatglm_cache(model_path: str) -> bool:
        if not is_glm_model(model_path):
            return False
        snapshot = os.path.basename(os.path.normpath(model_path))
        cache_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
        module_path = os.path.join(cache_root, snapshot, "modeling_chatglm.py")
        if not os.path.exists(module_path):
            return False
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
        marker = "\n        if len(presents) == 0:\n"
        guard = "\n        if presents is None:\n            presents = []\n        if len(presents) == 0:\n"
        if marker in content and "presents is None" not in content:
            content = content.replace(marker, guard)
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[INFO] Patched ChatGLM cache guard in {module_path}")
            return True
        return False
    def load_llm(model_path, **common_kwargs):
        try:
            llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                **common_kwargs,
            )
            return llm
        except ValueError as exc:
            if not (is_glm_model(model_path) and "config_class" in str(exc)):
                raise

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            auto_map = getattr(config, "auto_map", {}) or {}
            class_ref = auto_map.get("AutoModelForCausalLM")
            if class_ref is None:
                raise

            model_class = get_class_from_dynamic_module(class_ref, model_path)
            llm = model_class.from_pretrained(model_path, **common_kwargs)
            return llm

    common_kwargs = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    patched = maybe_patch_chatglm_cache(args.model_name)
    model = load_llm(args.model_name, **common_kwargs)
    if not patched:
        maybe_patch_chatglm_cache(args.model_name)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    model.config.use_cache = True

    enable_duo_attention_training(
        model,
        args.sink_size,
        args.recent_size,
        args.max_length,
        initial_value=args.initial_value,
        enable_ulysses_attention=True,
        streaming_attn_implementation=args.streaming_attn_implementation,
    )

    model = get_model_backbone(model)

    for param in model.parameters():
        param.requires_grad = False

    num_attn_heads = 0
    for name, param in model.named_parameters():
        if "full_attention_heads" in name:
            param.requires_grad = True
            num_attn_heads += param.numel()

    setup()

    torch.cuda.set_device(local_rank)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    model_type = str(getattr(config, "model_type", "")).lower()
    apply_activation_checkpointing(model)

    # mesh = None
    mesh = DeviceMesh(device_type="cuda", mesh=[i for i in range(world_size)])

    apply_fsdp(
        model,
        mesh,
        mp_policy,
        modules_to_shard=get_modules_to_shard(model),
    )

    if rank == 0:
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(
                    f"Trainable parameter: {name} with shape {param.shape}, dtype {param.dtype}, device {param.device}"
                )

    haystack_dataset = get_dataset(args.dataset_name, split="train")

    if args.dataset_format == "multiple_passkey":
        train_dataset = MultiplePasskeyRetrievalDataset(
            haystack_dataset,
            tokenizer,
            max_length=args.max_length,
            min_depth_ratio=args.min_needle_depth_ratio,
            max_depth_ratio=args.max_needle_depth_ratio,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
            depth_ratio_num_intervals=args.depth_ratio_num_intervals,
            num_passkeys=args.num_passkeys,
        )
    else:
        raise ValueError(f"Invalid dataset format: {args.dataset_format}")

    train_dataloader = get_supervised_dataloader(
        train_dataset, tokenizer, args.batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    warmup_steps = max(args.num_steps // 5, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1,
            max((step + 1) / warmup_steps, 0.1),
            max((args.num_steps - step) / warmup_steps, 0.1),
        ),
    )
    if rank == 0:
        experiment_config = vars(args)
        if not args.disable_wandb:
            wandb.init(project="DuoAttention", config=experiment_config)
            if args.exp_name is not None:
                wandb.run.name = args.exp_name

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                json.dump(experiment_config, f)

    # if resume and link exists, load the latest state
    if args.resume and os.path.exists(
        os.path.join(
            args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
        )
    ):
        # load the latest state in the output_dir
        state = torch.load(
            os.path.join(
                args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
            )
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        full_attention_heads = load_full_attention_heads(
            args.output_dir, filename="full_attention_heads_latest.tsv"
        )
        set_full_attention_heads(model, full_attention_heads)
        resume_step = state["global_step"]
        print(f"Resuming from step {resume_step}")
    else:
        resume_step = -1

    train(
        args,
        model,
        rank,
        world_size,
        train_dataloader,
        optimizer,
        scheduler,
        resume_step,
    )

    full_attention_heads = get_full_attention_heads(model)
    full_attention_heads = [
        h.full_tensor() if hasattr(h, "full_tensor") else h for h in full_attention_heads
    ]

    if rank == 0:
        print("Training finished")
        if args.output_dir is not None:
            full_attention_heads_list = full_attention_heads_to_list(
                full_attention_heads
            )
            # save the full attention heads as tsv
            save_full_attention_heads(
                full_attention_heads_list,
                os.path.join(args.output_dir, "full_attention_heads.tsv"),
            )

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)
