"""Suppress attention to source-statement tokens ONLY while decoding the
output, leaving the prompt prefill untouched.

Rationale: the existing source-boost experiments forced more attention onto
the statement and failed. The opposite intervention — letting the model
encode the statement normally during prefill but lowering attention to it
while generating the CoT/answer — has not been tried. The hypothesis here
is task-vs-data: lowering source attention at decode time should let the
model treat the proposition as background data rather than content to
dissect.

Detection of decode vs prefill: with HF KV-cached generate(), the prefill
forward has T_q = prompt_len > 1; every subsequent decode step has T_q = 1.
Bias is applied only when T_q == 1.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from src.utils.models import set_model, get_layers, is_qwen3, load_model_eager
import src.utils.models as ml
from src.utils.general import PROJECT_ROOT, parse_answer, load_kable_items


# SDPA fast path bypasses our hook (kwargs["attention_mask"] is None for the
# default causal case), so we stay with eager. Verified empirically on
# gemma-3-4b: identical outputs across alphas under SDPA.
load_model_sdpa = load_model_eager


def build_token_mask(prompt_text, raw_sentence, tok):
    """Tokenize prompt with offsets; mark tokens inside any occurrence of
    raw_sentence as source content (True). Returns input_ids, mask, spans."""
    enc = tok(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    spans = []
    start = 0
    while True:
        pos = prompt_text.find(raw_sentence, start)
        if pos < 0:
            break
        spans.append((pos, pos + len(raw_sentence)))
        start = pos + 1

    mask = np.zeros(len(ids), dtype=bool)
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        for ss, ee in spans:
            if s >= ss and e <= ee:
                mask[i] = True
                break
    return ids, mask, spans


class DecodeOnlySuppressHooks:
    """Adds a negative bias to source-token columns of the attention mask,
    only on forward passes where T_q == 1 (i.e., decode steps with KV cache).
    """
    def __init__(self, model, source_mask_np, alpha, max_total_len=4096):
        self.alpha = alpha
        self.device = next(model.parameters()).device
        prompt_len = len(source_mask_np)
        dtype = torch.bfloat16
        boost = torch.zeros(max_total_len, device=self.device, dtype=dtype)
        idx = torch.from_numpy(np.where(source_mask_np)[0]).long().to(self.device)
        boost[idx] = float(alpha)
        self.boost = boost
        self.prompt_len = prompt_len
        self.handles = []
        for layer in get_layers(model):
            attn = getattr(layer, "self_attn", layer)
            self.handles.append(
                attn.register_forward_pre_hook(self._pre, with_kwargs=True))

    def _pre(self, module, args, kwargs):
        am = kwargs.get("attention_mask")
        if am is None:
            return args, kwargs
        T_q = am.shape[-2]
        if T_q != 1:
            return args, kwargs  # prefill — leave alone
        T_k = am.shape[-1]
        if T_k > self.boost.shape[0]:
            extra = T_k - self.boost.shape[0]
            self.boost = torch.cat([self.boost,
                                     torch.zeros(extra, device=self.boost.device,
                                                 dtype=self.boost.dtype)])
        b = self.boost[:T_k].view(1, 1, 1, T_k).to(dtype=am.dtype, device=am.device)
        kwargs["attention_mask"] = am + b
        return args, kwargs

    def remove(self):
        for h in self.handles:
            h.remove()


def run_alpha(model, tok, items, alpha, max_new_tokens):
    chat_kwargs = {"enable_thinking": False} if is_qwen3(ml.MODEL_ID) else {}
    device = next(model.parameters()).device
    n_correct = 0; n_parsed = 0; n_skipped = 0
    saved = []
    for item in tqdm(items, desc=f"α={alpha}"):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": item["query"]}],
            tokenize=False, add_generation_prompt=True, **chat_kwargs,
        )
        try:
            ids, source_mask, spans = build_token_mask(prompt, item["raw_sentence"], tok)
        except Exception:
            n_skipped += 1; continue
        if not spans or source_mask.sum() == 0:
            n_skipped += 1; continue
        hooks = DecodeOnlySuppressHooks(model, source_mask, alpha,
                                         max_total_len=len(ids)+max_new_tokens+10)
        try:
            input_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
        finally:
            hooks.remove()
        gen = out[0, len(ids):]
        text = tok.decode(gen, skip_special_tokens=True)
        ans = parse_answer(text)
        if ans is None:
            n_skipped += 1
        else:
            n_parsed += 1
            if ans == item.get("gold", "A"): n_correct += 1
        saved.append({"idx": item["idx"], "alpha": alpha,
                      "gold": item.get("gold", "A"),
                      "answer": ans, "response_len": int(len(gen)),
                      "response": text[:300]})
    return {"alpha": alpha, "n_items": len(items), "n_parsed": n_parsed,
            "n_correct": n_correct, "acc": n_correct / max(n_parsed, 1),
            "items": saved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-3b")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, -0.5, -1.0, -2.0, -4.0])
    ap.add_argument("--n_items", type=int, default=50)
    ap.add_argument("--item_start", type=int, default=0,
                    help="skip this many shuffled items before taking n_items (for incremental runs)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--item_type", choices=["false", "factual"], default="false")
    ap.add_argument("--task", choices=["confirmation", "verification"],
                    default="confirmation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_model(args.model)
    tok, model = load_model_sdpa()
    print(f"loaded {args.model}", flush=True)

    items = load_kable_items(args.item_type, args.task)
    rng = random.Random(args.seed); rng.shuffle(items)
    items = items[args.item_start:args.item_start + args.n_items]
    print(f"sampling {len(items)} {args.item_type}-X items (task={args.task}, "
          f"gold={items[0]['gold']})", flush=True)

    results = []
    for alpha in args.alphas:
        r = run_alpha(model, tok, items, alpha, args.max_new_tokens)
        print(f"  α={alpha:+5.2f} (source-suppress decode-only): "
              f"parsed={r['n_parsed']}/{r['n_items']}  "
              f"correct(A)={r['n_correct']}  acc={r['acc']:.3f}", flush=True)
        results.append(r)

    out_path = args.out or (
        f"results/attention_suppress_decode/"
        f"{args.model}_{args.task}_{args.item_type}.json")
    out_path = os.path.join(PROJECT_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "n_items": args.n_items,
                   "alphas": args.alphas, "results": results}, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)
    print(f"\n{'α':>6}  {'parsed':>7}  {'correct(A)':>10}  {'acc':>6}")
    for r in results:
        print(f"{r['alpha']:>6}  {r['n_parsed']:>7}  {r['n_correct']:>10}  {r['acc']:>6.3f}")


if __name__ == "__main__":
    main()
