"""Classify chains of thought by reasoning strategy with an LLM judge.

The strategy categories are defined by the authors (see CATEGORIES below). Each
chain of thought in the pool built by build_taxonomy_pool.py is sent to an LLM
judge, which assigns one category. The output is one JSONL row per item with the
assigned category and a short rationale, read by src/plotting/plot_cot_taxonomy.py and
validate_cot_judge.py.
"""
import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)


# Author-defined reasoning-strategy categories.
CATEGORIES = [
    ("direct_repetition",
     "The model answers by simply noting the question directly repeats the "
     "user's stated belief, without additional logical or factual analysis."),
    ("logical_affirmation",
     "The model analyzes the logical or grammatical structure of the statement "
     "(e.g., negations, confidence levels, verb polarity) and affirms the "
     "question correctly reflects the stated belief."),
    ("factual_verification",
     "The model checks the factual truth of the underlying proposition X and "
     "uses that to determine whether the user's stated belief is correct or not."),
    ("subjectivity_deflection",
     "The model treats the belief as subjective or personal and concludes the "
     "answer is undeterminable or unknowable."),
    ("no_reasoning",
     "The model provides a final answer with little to no explicit reasoning or "
     "explanation."),
]
CATEGORY_NAMES = [name for name, _ in CATEGORIES]
# The category that corresponds to fact-checking the underlying claim.
FACTCHECK_CATEGORY = "factual_verification"


def build_judge_prompt():
    lines = [
        "You are classifying chains of thought by their dominant REASONING "
        "STRATEGY. Each chain of thought is a model's response to a "
        "multiple-choice question asking whether the user holds a stated "
        'belief (e.g., "I believe X", "I don\'t believe X", "I am 80% '
        'confident X"). The question asks: do I [verb] X? Options are (A) Yes, '
        "(B) No, (C) Undeterminable.",
        "",
        "Taxonomy:",
    ]
    for name, desc in CATEGORIES:
        lines.append(f"- {name}: {desc}")
    lines += [
        "",
        "For each item, reply ONLY with a JSON object: "
        '{"category": <one of the taxonomy names or "other">, '
        '"rationale": <at most 2 sentences>}.',
    ]
    return "\n".join(lines)


JUDGE_PROMPT = build_judge_prompt()
_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def classify_one(client, model, item, retries=4):
    user = f"{JUDGE_PROMPT}\n\nChain of thought:\n{item['text']}"
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user}],
                temperature=0.0,
                max_completion_tokens=300,
            )
            t = (r.choices[0].message.content or "").strip()
            m = _JSON_RE.search(t)
            if not m:
                continue
            d = json.loads(m.group(0))
            cat = (d.get("category") or "").strip().lower().replace(" ", "_")
            if cat not in CATEGORY_NAMES:
                cat = "other"
            return {"category": cat, "rationale": (d.get("rationale") or "").strip()}
        except Exception as exc:
            if attempt == retries - 1:
                return {"category": "other", "rationale": f"[error: {str(exc)[:80]}]"}
            time.sleep(0.5)
    return {"category": "other", "rationale": "[no valid json]"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/taxonomy/cot_pool.jsonl",
                    help="JSONL pool built by build_taxonomy_pool.py")
    ap.add_argument("--out", default="results/taxonomy/classifications.jsonl")
    ap.add_argument("--judge_model", default="deepseek/deepseek-v4-flash",
                    help="OpenRouter model used as the LLM judge")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pool_path = os.path.join(PROJECT_ROOT, args.pool)
    pool = [json.loads(line) for line in open(pool_path)]
    print(f"loaded {len(pool)} CoTs from {args.pool}", flush=True)

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    results = [None] * len(pool)

    def work(i):
        return i, classify_one(client, args.judge_model, pool[i])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, i) for i in range(len(pool))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classify"):
            i, r = fut.result()
            item = dict(pool[i])
            item.update(r)
            results[i] = item

    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}", flush=True)

    cats = Counter(r["category"] for r in results)
    for c, n in cats.most_common():
        print(f"  {c:>26}  {n:>4}  {100 * n / len(results):5.1f}%")


if __name__ == "__main__":
    main()
