"""Shared model loader and registry.

Extracted from the (now-archived) persona-vector pipeline so downstream
probing / patching / evaluation scripts don't depend on persona_vector_two.py.

Usage:
    from src.utils.models import (
        MODELS, set_model, load_model, get_layers, is_qwen3,
    )
    import src.utils.models as ml
    ...
    set_model("qwen-3.5-4b")
    tok, model = load_model()
    chat_kwargs = {"enable_thinking": False} if is_qwen3(ml.MODEL_ID) else {}
"""
import torch

MODELS = {
    "gemma-3-12b": {"hf": "google/gemma-3-12b-it",  "or": "google/gemma-3-12b-it"},
    "gemma-3-4b":  {"hf": "google/gemma-3-4b-it",   "or": "google/gemma-3-4b-it"},
    "llama-3.2-3b": {"hf": "meta-llama/Llama-3.2-3B-Instruct",
                     "or": "meta-llama/llama-3.2-3b-instruct"},
    "llama-3.1-8b": {"hf": "meta-llama/Llama-3.1-8B-Instruct",
                     "or": "meta-llama/llama-3.1-8b-instruct"},
    "qwen-3.5-9b": {"hf": "Qwen/Qwen3.5-9B", "or": "qwen/qwen3.5-9b"},
    "qwen-3.5-4b": {"hf": "Qwen/Qwen3.5-4B", "or": "qwen/qwen3.5-4b"},
}

QWEN_THINK_PREFILL = "<think>\n</think>\n\n"


def is_qwen3(model_or_name: str) -> bool:
    name = (model_or_name or "").lower()
    return "qwen3" in name or "qwen-3" in name


# Globals set by set_model() before any phase runs.
MODEL_ID = None
OR_MODEL = None


def set_model(name):
    global MODEL_ID, OR_MODEL
    cfg = MODELS[name]
    MODEL_ID = cfg["hf"]
    OR_MODEL = cfg["or"]


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


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"trust_remote_code": True} if is_qwen3(MODEL_ID) else {}
    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", **kwargs,
    )
    model.eval()
    return tok, model


def text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def load_model_eager():
    """Load model with eager attention so attention-mask hooks and
    output_attentions work (SDPA's fast path bypasses both)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"trust_remote_code": True} if is_qwen3(MODEL_ID) else {}
    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
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
