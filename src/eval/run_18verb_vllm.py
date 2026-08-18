"""Run the KABLE 18-verb evaluation with vLLM for one model.

Same output format as src/run_18verb_local.py — one JSON per prompt under
`logs/<prompt_template>/<verb>/<model_safe>/<i>.json` — but uses vLLM's
batched generation for ~5–10x speedup.
"""
import argparse
import json
import os
import time

from vllm import LLM, SamplingParams

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.models import MODELS, is_qwen3
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


def run_one_verb(llm, model_name, hf_id, verb, items, prompt_template,
                 output_base, max_new_tokens, model_safe, enable_thinking):
    verb_map = get_verb_mappings()[verb]
    out_dir = os.path.join(output_base, prompt_template, verb, model_safe)
    os.makedirs(out_dir, exist_ok=True)
    done = existing_indices(out_dir)
    todo = [(i, ex) for i, ex in enumerate(items) if i not in done]
    if not todo:
        print(f"  {verb}: already complete", flush=True)
        return
    print(f"  {verb}: {len(todo)}/{len(items)} to generate", flush=True)

    conversations = []
    chat_kwargs = {}
    if enable_thinking is False:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    for i, ex in todo:
        conv = [{"role": "user", "content": map_query(ex["query"], verb_map)}]
        conversations.append(conv)

    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    t0 = time.time()
    outs = llm.chat(conversations, sampling_params=sp, use_tqdm=True,
                    **chat_kwargs)
    print(f"  {verb}: generated in {time.time() - t0:.1f}s", flush=True)

    for (i, ex), out in zip(todo, outs):
        record = {
            "index": i,
            "verb_tested": verb,
            "model": model_name,
            "prompt_used": map_query(ex["query"], verb_map),
            "original_data": {k: ex[k] for k in ex if k != "query"},
            "ground_truth": ex["answer"],
            "model_output": out.outputs[0].text,
        }
        with open(os.path.join(out_dir, f"{i}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--verbs", default="all")
    ap.add_argument("--prompt_template", default="original")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--dataset_file",
                    default="data/confirmation-of-first-person-belief.jsonl")
    ap.add_argument("--output_base", default="logs")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    hf_id = cfg["hf"]
    model_safe = cfg["or"].replace("/", "_")

    items = load_kable(args.dataset_file)
    verbs = list(get_verb_mappings().keys()) if args.verbs == "all" else \
        args.verbs.split(",")

    print(f"model={args.model}  hf={hf_id}", flush=True)
    print(f"verbs ({len(verbs)}): {verbs}", flush=True)
    print(f"out: {args.output_base}/{args.prompt_template}/<verb>/"
          f"{model_safe}/", flush=True)

    enable_thinking = False if is_qwen3(hf_id) else None
    llm_kwargs = dict(
        model=hf_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        dtype="bfloat16",
    )
    if is_qwen3(hf_id):
        llm_kwargs["trust_remote_code"] = True
        # Qwen3.5 has architecture Qwen3_5ForConditionalGeneration (multimodal),
        # which vLLM doesn't support. Override to Qwen3ForCausalLM to use the
        # text-only Qwen3 path — works because the underlying decoder is
        # structurally compatible with Qwen3.
        llm_kwargs["hf_overrides"] = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "max_window_layers": 0,
            "use_sliding_window": False,
        }
    print(f"loading vLLM with kwargs={ {k: v for k, v in llm_kwargs.items() if k != 'model'} }",
          flush=True)
    llm = LLM(**llm_kwargs)

    for verb in verbs:
        print(f"\n=== verb={verb} ===", flush=True)
        run_one_verb(llm, args.model, hf_id, verb, items, args.prompt_template,
                     args.output_base, args.max_new_tokens, model_safe,
                     enable_thinking)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
