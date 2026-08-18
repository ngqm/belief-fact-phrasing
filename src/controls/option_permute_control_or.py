"""Answer position control via OpenRouter (matches the main behavioral eval).

Permute the option ORDER so "Yes" sits at (A), (B), or (C), remap the gold
letter, and re-measure the accuracy gap between factual and false claims and
the gain from the no_fact_check template. If both persist when "Yes" leaves
(A), neither is an artifact of answer position. Served through OpenRouter
(temperature 0), the same way the paper's main evaluation is served; no local
GPU is needed.

Companion to src/controls/option_permute_control.py (local HF), which already produced
qwen-3.5-4b (local-only model). This covers the 9 OpenRouter-served models.
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.general import get_model_response, extract_mcq_answer, get_all_models, NO_FACT_CHECK

load_dotenv(override=True)
CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"))
DATA = "data/confirmation-of-first-person-belief.jsonl"

ORIG_BLOCK = "(A) Yes\n(B) No\n(C) Undeterminable"
PERMS = {
    "orig": ("(A) Yes\n(B) No\n(C) Undeterminable", "(A)"),
    "yesB": ("(A) No\n(B) Yes\n(C) Undeterminable", "(B)"),
    "yesC": ("(A) No\n(B) Undeterminable\n(C) Yes", "(C)"),
}

OR2SHORT = {
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
    "meta-llama/llama-3.2-3b-instruct": "llama-3.2-3b",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
    "google/gemma-3-4b-it": "gemma-3-4b",
    "google/gemma-3-12b-it": "gemma-3-12b",
    "google/gemma-3-27b-it": "gemma-3-27b",
    "qwen/qwen3.5-9b": "qwen-3.5-9b",
    "qwen/qwen3.5-27b": "qwen-3.5-27b",
    "qwen/qwen3.5-35b-a3b": "qwen-3.5-35b-a3b",
}
SHORT2OR = {v: k for k, v in OR2SHORT.items()}


def load_confirm(itype, n, start=0):
    items = []
    for line in open(DATA):
        d = json.loads(line)
        if d["type"] == itype:
            items.append(d["query"])
    return items[start:start + n]


def permute(query, block):
    return query.replace(ORIG_BLOCK, block)


def add_nofc(query):
    return query.replace("\nAnswer:", f"\n\n{NO_FACT_CHECK}\nAnswer:")


def one(model_or, query):
    comp = get_model_response(CLIENT, model_or, query)
    if comp is None:
        return None
    try:
        return extract_mcq_answer(comp.choices[0].message.content or "")
    except Exception:
        return None


def run_cell(model_or, queries, gold, workers):
    ans = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, model_or, q): i for i, q in enumerate(queries)}
        for f in as_completed(futs):
            ans[futs[f]] = f.result()
    parsed = sum(1 for a in ans if a)
    n_ok = sum(1 for a in ans if a == gold)
    return n_ok / max(len(queries), 1), parsed


def run_model(model_or, short, n, workers):
    out = f"results/option_permute/{short}.json"
    results = {}
    for perm, (block, gold) in PERMS.items():
        for prompt in ["orig", "nofc"]:
            for itype in ["factual", "false"]:
                qs = [permute(q, block) for q in load_confirm(itype, n)]
                if prompt == "nofc":
                    qs = [add_nofc(q) for q in qs]
                acc, par = run_cell(model_or, qs, gold, workers)
                results[f"{perm}_{prompt}_{itype}"] = {"acc": acc, "parsed": par, "n": len(qs)}
                print(f"  {short:16s} {perm:5s} {prompt:4s} {itype:8s}: acc={acc:.3f} parsed={par}/{len(qs)}", flush=True)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                json.dump({"model": short, "results": results}, open(out, "w"), indent=2)
    print(f"wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="short names; default = all 9 OpenRouter models")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.models:
        targets = [(SHORT2OR[m], m) for m in args.models]
    else:
        targets = [(orid, OR2SHORT[orid]) for orid in get_all_models() if orid in OR2SHORT]
    for orid, short in targets:
        print(f"=== {short} ({orid}) ===", flush=True)
        run_model(orid, short, args.n, args.workers)


if __name__ == "__main__":
    main()
