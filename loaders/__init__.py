from __future__ import annotations

from importlib import import_module

from benchmarks.common import model_path


METHOD_LOADERS = {
    "full_attention": "full_attention_load",
    "topk": "pqcache_load",
    "topk32": "pqcache_load",
    "topp": "pqcache_load",
    "topp32": "pqcache_load",
    "pqcache": "pqcache_load",
    "clusterkv": "clusterkv_load",
    "h2o": "h2o_load",
    "keyformer": "keyformer_load",
    "snapkv": "snapkv_load",
    "streaming": "streaming_load",
    "quest": "quest_load",
    "sparq": "sparq_load",
    "xattention": "xattention_load",
    "flexprefill": "flexprefill_load",
    "minference": "minference_load",
    "retroinfer": "retroinfer_load",
    "magicpig": "magicpig_load",
    "headkv": "headkv_load",
    "adakv": "adakv_load",
    "cakekv": "cakekv_load",
    "duo-attention": "duo_attention_load",
    "pyramidkv": "pyramidkv_load",
}


def load_model_and_tokenizer(args):
    """Load the model implementation selected by the unified method name."""
    args.model_path = model_path(args.model)
    try:
        loader_name = METHOD_LOADERS[args.method]
    except KeyError as error:
        raise ValueError(f"No loader is registered for method: {args.method}") from error

    loader = import_module(f"loaders.{loader_name}")
    load = getattr(loader, "load_model_and_tokenizer", None)
    if load is None:
        raise NotImplementedError(
            f"loaders/{loader_name}.py must define load_model_and_tokenizer(args) "
            f"for method '{args.method}'."
        )
    return load(args)
