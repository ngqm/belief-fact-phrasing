"""Second-pass agreement check for the CoT strategy labels.

Relabels a stratified sample of the primary judge's classifications with an
independent second judge, then reports Cohen's kappa and raw agreement. In the
paper the second pass was reviewed by the authors. The categories and the judge
prompt are the author-defined ones in classify_cot_strategies.py.
"""
import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from src.taxonomy.classify_cot_strategies import (
    JUDGE_PROMPT, CATEGORY_NAMES, PROJECT_ROOT,
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)


def relabel_one(client, model, item, retries=2):
    user = f"{JUDGE_PROMPT}\n\nChain of thought:\n{item['text']}"
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": user}],
            )
            t = (r.choices[0].message.content or "").strip()
            s, e = t.find("{"), t.rfind("}")
            if s == -1 or e == -1:
                raise ValueError("no json")
            cat = json.loads(t[s:e + 1]).get("category") or "other"
            return cat if cat in CATEGORY_NAMES else "other"
        except Exception:
            if attempt == retries:
                return "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifications",
                    default="results/taxonomy/classifications.jsonl")
    ap.add_argument("--judge", required=True,
                    help="Second, independent judge model (OpenRouter id)")
    ap.add_argument("--per_category", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/taxonomy/judge_agreement.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    path = os.path.join(PROJECT_ROOT, args.classifications)
    labeled = [json.loads(line) for line in open(path)]
    rng = random.Random(args.seed)

    sample = []
    for c in CATEGORY_NAMES:
        pool = [it for it in labeled if it.get("category") == c]
        rng.shuffle(pool)
        sample.extend(pool[:args.per_category])
    print(f"sampled {len(sample)}: {Counter(s['category'] for s in sample)}")

    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])

    second = [None] * len(sample)

    def work(i):
        return i, relabel_one(client, args.judge, sample[i])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, i) for i in range(len(sample))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="second pass"):
            i, cat = fut.result()
            second[i] = cat

    pairs = [(sample[i]["category"], second[i]) for i in range(len(sample))
             if second[i] != "error"]
    primary = [p[0] for p in pairs]
    other = [p[1] for p in pairs]
    labels = CATEGORY_NAMES + ["other"]
    kappa = cohen_kappa_score(primary, other, labels=labels)
    agree = sum(1 for x, y in zip(primary, other) if x == y) / len(pairs)
    cm = confusion_matrix(primary, other, labels=labels)

    print(f"\nCohen's kappa = {kappa:.3f}")
    print(f"Raw agreement = {agree:.3f} "
          f"({sum(1 for x, y in zip(primary, other) if x == y)}/{len(pairs)})")

    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({
        "n_pairs": len(pairs),
        "kappa": kappa,
        "raw_agreement": agree,
        "categories": labels,
        "judge_second": args.judge,
        "per_category_sampled": args.per_category,
        "seed": args.seed,
        "confusion_matrix": cm.tolist(),
    }, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
