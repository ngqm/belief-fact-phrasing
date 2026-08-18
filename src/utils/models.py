"""Model registry and loader.

    from src.utils.models import MODELS, load_model, get_layers, is_qwen3
    tok, model = load_model(MODELS["qwen-3.5-4b"]["hf"])
"""
import torch

from src.utils.general import is_qwen3

MODELS = {
    "gemma-3-12b": {"hf": "google/gemma-3-12b-it",  "or": "google/gemma-3-12b-it"},
    "gemma-3-4b":  {"hf": "google/gemma-3-4b-it",   "or": "google/gemma-3-4b-it"},
    "llama-3.2-3b": {"hf": "meta-llama/Llama-3.2-3B-Instruct",
                     "or": "meta-llama/llama-3.2-3b-instruct"},
    "llama-3.1-8b": {"hf": "meta-llama/Llama-3.1-8B-Instruct",
                     "or": "meta-llama/llama-3.1-8b-instruct"},
    "qwen-3.5-9b": {"hf": "Qwen/Qwen3.5-9B", "or": "qwen/qwen3.5-9b"},
    "qwen-3.5-4b": {"hf": "Qwen/Qwen3.5-4B", "or": "qwen/qwen3.5-4b"},
    "gemma-3-27b": {"hf": "google/gemma-3-27b-it", "or": "google/gemma-3-27b-it"},
    "qwen-3.5-27b": {"hf": "Qwen/Qwen3.5-27B", "or": "qwen/qwen3.5-27b"},
    "qwen-3.5-35b-a3b": {"hf": "Qwen/Qwen3.5-35B-A3B", "or": "qwen/qwen3.5-35b-a3b"},
    "llama-3.3-70b": {"hf": "meta-llama/Llama-3.3-70B-Instruct",
                      "or": "meta-llama/llama-3.3-70b-instruct"},
}


def get_layers(model):
    """Locate the decoder layer ModuleList across HF model families and versions."""
    paths = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.text_model.layers,
        lambda m: m.transformer.h,
    ]
    for fn in paths:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(f"can't find decoder layers on {type(model).__name__}")


def load_model(model_id):
    """Load the tokenizer and model in bfloat16 with automatic device mapping.

    Args:
        model_id: the HuggingFace model id to load.

    Returns:
        the (tokenizer, model) pair.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"trust_remote_code": True} if is_qwen3(model_id) else {}
    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", **kwargs,
    )
    model.eval()
    return tok, model


def text_config(model):
    """Return the model's text config, or the config itself if there is no text_config."""
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def load_model_eager(model_id):
    """Load the tokenizer and model with eager attention.

    Eager attention keeps attention-mask hooks and output_attentions working,
    which SDPA's fast path bypasses.

    Args:
        model_id: the HuggingFace model id to load.

    Returns:
        the (tokenizer, model) pair.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"trust_remote_code": True} if is_qwen3(model_id) else {}
    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", **kwargs,
    )
    model.eval()
    return tok, model


def build_prompt(query, tok, is_qwen3_flag):
    """Apply the chat template to a bare user query."""
    chat_kwargs = {"enable_thinking": False} if is_qwen3_flag else {}
    return tok.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True, **chat_kwargs,
    )
