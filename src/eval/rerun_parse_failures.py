"""Re-run items where the visible model_output was empty (or the MCQ parse
failed) at a larger max_tokens budget. Overwrites the JSON in place."""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from src.utils.general import extract_mcq_answer

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def needs_rerun(d):
    mo = (d.get("model_output") or "").strip()
    if not mo:
        return True
    return extract_mcq_answer(mo) == ""


def rerun_one(client, model, path, max_tokens):
    d = json.load(open(path))
    prompt = d["prompt_used"]
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return path, False, str(e)
    d["model_output"] = completion.choices[0].message.content
    d["full_api_response"] = completion.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return path, needs_rerun(d) is False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="OpenRouter ID, e.g. deepseek/deepseek-v4-flash")
    ap.add_argument("--model_dir", required=True,
                    help="On-disk dir name, e.g. deepseek_deepseek-v4-flash")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--templates", nargs="+",
                    default=["original", "no_fact_check",
                             "may_or_may_not_fact_check", "must_fact_check"])
    args = ap.parse_args()

    failures = []
    for tmpl in args.templates:
        tmpl_dir = os.path.join(PROJECT_ROOT, "logs", tmpl)
        if not os.path.isdir(tmpl_dir):
            continue
        for verb in sorted(os.listdir(tmpl_dir)):
            mdir = os.path.join(tmpl_dir, verb, args.model_dir)
            if not os.path.isdir(mdir):
                continue
            for f in glob.glob(os.path.join(mdir, "*.json")):
                d = json.load(open(f))
                if needs_rerun(d):
                    failures.append(f)
    print(f"To rerun: {len(failures)} items at max_tokens={args.max_tokens}",
          flush=True)
    if not failures:
        return

    client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)
    n_ok = n_err = n_still_bad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(rerun_one, client, args.model, p, args.max_tokens): p
                for p in failures}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            path, ok, err = fut.result()
            if err is not None:
                n_err += 1
                continue
            if ok:
                n_ok += 1
            else:
                n_still_bad += 1
    print(f"\nfixed={n_ok}  still_empty_or_unparseable={n_still_bad}  "
          f"errored={n_err}", flush=True)


if __name__ == "__main__":
    main()
