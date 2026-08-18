"""Run the KABLE 18-verb evaluation with batched transformers generation.

A plain transformers fallback for machines without a vLLM install.
Batches prompts together inside model.generate() for ~5–10x speedup over
single-prompt generation.
"""
import argparse
import json
import os
import time

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.models import (
    MODELS, set_model, load_model, is_qwen3,
)
import src.utils.models as pv2
from src.utils.general import get_verb_mappings


def load_kable(dataset_file):
    return [json.loads(l) for l in open(dataset_file)]


def map_query(query, verb_map):
    q = query.replace("Do I believe", verb_map["question_form"])
    q = q.replace("I believe", verb_map["statement_form"])
    return q


def existing_indices(out_dir):
    if not os.path.isdir(out_dir):
        return set()
    return {int(f.split(".")[0]) for f in os.listdir(out_dir)
            if f.endswith(".json")}


def load_system_prompt(prompt_template):
    import json as _json
    with open("data/prompt_templates.json") as f:
        return _json.load(f).get(prompt_template, "")


def run_one_verb(model_name, verb, tok, model, items, prompt_template,
                 output_base, max_new_tokens, chat_kwargs, model_safe,
                 batch_size, system_prompt=""):
    verb_map = get_verb_mappings()[verb]
    out_dir = os.path.join(output_base, prompt_template, verb, model_safe)
    os.makedirs(out_dir, exist_ok=True)
    done = existing_indices(out_dir)
    todo = [(i, ex) for i, ex in enumerate(items) if i not in done]
    if not todo:
        print(f"  {verb}: already complete", flush=True)
        return

    device = next(model.parameters()).device
    # Left-pad for generation.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"  {verb}: {len(todo)} to generate, batch_size={batch_size}",
          flush=True)
    t0 = time.time()
    for bs in tqdm(range(0, len(todo), batch_size),
                   desc=verb, leave=False):
        batch = todo[bs:bs + batch_size]
        prompts = []
        for _, ex in batch:
            q = map_query(ex["query"], verb_map)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": q})
            prompt_text = tok.apply_chat_template(
                messages,
                tokenize=False, add_generation_prompt=True, **chat_kwargs,
            )
            prompts.append(prompt_text)
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        # Slice off the prompt portion and decode each.
        for (i, ex), input_ids, full_ids in zip(
                batch, inputs["input_ids"], out_ids):
            gen_part = full_ids[input_ids.shape[0]:]
            resp = tok.decode(gen_part, skip_special_tokens=True)
            q = map_query(ex["query"], verb_map)
            record = {
                "index": i,
                "verb_tested": verb,
                "model": model_name,
                "prompt_used": q,
                "original_data": {k: ex[k] for k in ex if k != "query"},
                "ground_truth": ex["answer"],
                "model_output": resp,
            }
            with open(os.path.join(out_dir, f"{i}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  {verb}: {len(todo)} done in {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--verbs", default="all")
    ap.add_argument("--prompt_template", default="original")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dataset_file",
                    default="data/confirmation-of-first-person-belief.jsonl")
    ap.add_argument("--output_base", default="logs")
    args = ap.parse_args()

    set_model(args.model)
    tok, model = load_model()
    chat_kwargs = {"enable_thinking": False} if is_qwen3(pv2.MODEL_ID) else {}
    items = load_kable(args.dataset_file)

    verbs = list(get_verb_mappings().keys()) if args.verbs == "all" else \
        args.verbs.split(",")

    model_safe = MODELS[args.model]["or"].replace("/", "_")
    system_prompt = load_system_prompt(args.prompt_template)
    print(f"model={args.model}  hf={pv2.MODEL_ID}  "
          f"out_root={args.output_base}/{args.prompt_template}/<verb>/"
          f"{model_safe}/", flush=True)
    print(f"system_prompt: {system_prompt[:80]!r}", flush=True)
    print(f"verbs ({len(verbs)}): {verbs}", flush=True)

    for verb in verbs:
        print(f"\n=== verb={verb} ===", flush=True)
        run_one_verb(args.model, verb, tok, model, items, args.prompt_template,
                     args.output_base, args.max_new_tokens, chat_kwargs,
                     model_safe, args.batch_size, system_prompt=system_prompt)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
