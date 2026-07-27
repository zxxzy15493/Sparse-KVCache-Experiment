from benchmarks.common import method_defaults, parse_set

import torch    
import sys
from argparse import Namespace
from pathlib import Path

def load_model_and_tokenizer(args):
    """Load model with MInference sparse attention patch."""
    # Monkey-patch: GLM models define ChatGLMConfig in both modeling_chatglm.py
    # and configuration_chatglm.py, causing config_class mismatch in register().
    # Unify the model's config_class with the passed config class before the check.
    from transformers.models.auto.auto_factory import _BaseAutoModelClass
    from transformers import AutoTokenizer, AutoModelForCausalLM

    minference_root = Path(__file__).resolve().parents[1] / "methods" / "MInference" 
    if str(minference_root) not in sys.path:
        sys.path.insert(0, str(minference_root))

    from minference import MInference
    _orig_register = _BaseAutoModelClass.register.__func__

    def _patched_register(cls, config_class, model_class, exist_ok=False):
        if (
            hasattr(model_class, "config_class")
            and str(model_class.config_class) != str(config_class)
        ):
            model_class.config_class = config_class
        return _orig_register(cls, config_class, model_class, exist_ok=exist_ok)

    _BaseAutoModelClass.register = classmethod(_patched_register)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    finally:
        _BaseAutoModelClass.register = classmethod(_orig_register)

    model.eval()

    minference_patch = MInference(args.model_path)
    model = minference_patch(model)

    # full atten

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
