"""Build a stratified JSONL of CoT outputs for the strategy classifier.

Samples N per (model, verb, type) cell from logs/original/. The output is read
by classify_cot_strategies.py.
"""
import os
import sys
import json
import glob
import random
import argparse

from src.utils.general import get_full_cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs/original")
    ap.add_argument("--out", default="data/taxonomy/cot_pool.jsonl")
    ap.add_argument("--per_cell", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    written = 0
    cells_seen = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for verb in sorted(os.listdir(args.logs_dir)):
            verb_dir = os.path.join(args.logs_dir, verb)
            if not os.path.isdir(verb_dir):
                continue
            for model_dir in sorted(os.listdir(verb_dir)):
                model_path = os.path.join(verb_dir, model_dir)
                if not os.path.isdir(model_path):
                    continue
                files = glob.glob(os.path.join(model_path, "*.json"))
                by_type = {"factual": [], "false": []}
                for f in files:
                    try:
                        d = json.load(open(f))
                    except Exception:
                        continue
                    t = d.get("original_data", {}).get("type")
                    if t in by_type:
                        by_type[t].append((f, d))
                clean_model = (model_dir
                               .replace("google_", "")
                               .replace("meta-llama_", "")
                               .replace("qwen_", "")
                               .replace("-instruct", "")
                               .replace("-it", ""))
                for t, items in by_type.items():
                    cells_seen += 1
                    if not items:
                        continue
                    pick = rng.sample(items, min(args.per_cell, len(items)))
                    for f, d in pick:
                        text = get_full_cot(d)
                        if not text:
                            continue
                        # Use the filename (e.g. "742") rather than the
                        # in-file `index` field — older logs have a bug where
                        # the index field cycles 0..49 even though the 1000
                        # files in the cell hold 1000 distinct prompts.
                        file_idx = os.path.splitext(os.path.basename(f))[0]
                        item = {
                            "id": f"{clean_model}__{verb}__{t}__{file_idx}",
                            "text": text,
                            "model": clean_model,
                            "verb": verb,
                            "type": t,
                            "ground_truth": d.get("ground_truth", "").strip(),
                            "correct": d.get("ground_truth", "").strip() == _extract(text),
                        }
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                        written += 1

    print(f"cells: {cells_seen}  items written: {written}  → {args.out}")


def _extract(out: str) -> str:
    prefix = "So, the answer is"
    idx = out.find(prefix)
    if idx != -1:
        sec = out[idx + len(prefix):]
        for opt in ["(A)", "(B)", "(C)"]:
            if opt in sec:
                return opt
    for opt in ["(A)", "(B)", "(C)"]:
        if opt in out:
            return opt
    s = out.strip()
    if s in {"A", "B", "C"}:
        return f"({s})"
    return ""


if __name__ == "__main__":
    main()
