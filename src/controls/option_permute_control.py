"""Answer position control for the belief confirmation behavioral results.

KaBLE Task 5 gold is always (A) Yes, so raw accuracy could in principle
reflect an (A)-position bias. Permute the option ORDER so that "Yes" sits at
(A), (B), or (C), remap the gold letter, and re-measure. If the accuracy gap
between factual and false claims and the gain from the no_fact_check template
persist when "Yes" moves off (A), neither is an artifact of answer position.
"""
import argparse, json, os, torch
from src.utils.general import parse_answer, PROJECT_ROOT, NO_FACT_CHECK
import src.utils.models as ml
from src.utils.models import set_model, is_qwen3, load_model_eager, build_prompt

ORIG_BLOCK = "(A) Yes\n(B) No\n(C) Undeterminable"
PERMS = {
    "orig": ("(A) Yes\n(B) No\n(C) Undeterminable", "A"),
    "yesB": ("(A) No\n(B) Yes\n(C) Undeterminable", "B"),
    "yesC": ("(A) No\n(B) Undeterminable\n(C) Yes", "C"),
}
STOPS = ["\nuser", "\n\nuser", "<|im_start|>", "For each question, carefully"]


def load_confirm(item_type, n, start):
    path = os.path.join(PROJECT_ROOT, "data/confirmation-of-first-person-belief.jsonl")
    items = []
    for line in open(path):
        d = json.loads(line)
        if d["type"] == item_type:
            items.append(d["query"])
    return items[start:start + n]


def permute(query, perm_block):
    return query.replace(ORIG_BLOCK, perm_block)


def add_nofc(query):
    # append the no_fact_check instruction text after the trailing "Answer:"
    return query.replace("\nAnswer:", f"\n\n{NO_FACT_CHECK}\nAnswer:")


def run(model, tok, queries, gold, is_q3, max_new):
    dev = next(model.parameters()).device
    n_ok = n_par = 0
    for q in queries:
        ids = tok(build_prompt(q, tok, is_q3), return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id,
                                 stop_strings=STOPS, tokenizer=tok)
        ans = parse_answer(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        if ans: n_par += 1
        if ans == gold: n_ok += 1
    return n_ok / max(len(queries), 1), n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_items", type=int, default=100)
    ap.add_argument("--item_start", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_model(args.model)
    tok, model = load_model_eager()
    is_q3 = is_qwen3(ml.MODEL_ID)

    results = {}
    for perm, (block, gold) in PERMS.items():
        for prompt in ["orig", "nofc"]:
            for itype in ["factual", "false"]:
                qs = load_confirm(itype, args.n_items, args.item_start)
                qs = [permute(q, block) for q in qs]
                if prompt == "nofc":
                    qs = [add_nofc(q) for q in qs]
                acc, par = run(model, tok, qs, gold, is_q3, args.max_new_tokens)
                results[f"{perm}_{prompt}_{itype}"] = {"acc": acc, "parsed": par, "n": len(qs)}
                print(f"  {perm:5s} {prompt:4s} {itype:8s}: acc={acc:.3f} parsed={par}/{len(qs)}", flush=True)
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                json.dump({"model": args.model, "results": results}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
