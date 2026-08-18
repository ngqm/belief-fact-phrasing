"""Compute extended Table 1: per-verb-family breakdown of instruction-template
effects.

For each verb in:
  - Positive belief subset: believe, think, suppose, am_certain (existing Table 1)
  - Evidential family: vaguely_remember (new)
  - Confidence family: am_80_confident (new)
  - Negation family: seriously_doubt (existing spot-check)
and each of the 4 templates (original, no_fact_check, may_or_may_not_fact_check,
must_fact_check), compute Factual accuracy (%) and False accuracy (%) averaged
over the 10-model roster.

Prints both:
  (a) The original 4-verb-average table (sanity check)
  (b) A per-family extension table.
"""

import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANS_RE = re.compile(r"\(([ABCabc])\)")

MODELS = [
    "google_gemma-3-4b-it", "google_gemma-3-12b-it", "google_gemma-3-27b-it",
    "meta-llama_llama-3.1-8b-instruct", "meta-llama_llama-3.2-3b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "qwen_qwen3.5-4b", "qwen_qwen3.5-9b",
    "qwen_qwen3.5-27b", "qwen_qwen3.5-35b-a3b",
]

TEMPLATES = ["original", "no_fact_check", "may_or_may_not_fact_check",
             "must_fact_check"]

TEMPLATE_LABEL = {
    "original": "Original",
    "no_fact_check": "No fact-check",
    "may_or_may_not_fact_check": "May/may not fact-check",
    "must_fact_check": "Must fact-check",
}

POSITIVE_4 = ["believe", "think", "suppose", "am_certain"]
NEW_VERBS = ["vaguely_remember", "am_80_confident", "seriously_doubt"]


def extract_answer(text):
    """Find the last (A|B|C) in text — matches main.py's parser."""
    matches = ANS_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def parse_gold(gt):
    """ground_truth is stored as '(A)' or 'A' — normalize to letter."""
    m = ANS_RE.search(gt or "")
    return m.group(1).upper() if m else (gt.strip() if gt else None)


def load_cell(template, verb, model):
    """Returns dict: claim_type -> (n_correct, n_parsed, n_total) over all
    files in the cell."""
    d = os.path.join(PROJECT_ROOT, "logs", template, verb, model)
    if not os.path.isdir(d):
        return None
    counts = {"factual": [0, 0, 0], "false": [0, 0, 0]}
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn)) as f:
                rec = json.load(f)
        except Exception:
            continue
        ctype = rec.get("original_data", {}).get("type")
        if ctype not in counts:
            continue
        counts[ctype][2] += 1
        ans = extract_answer(rec.get("model_output", ""))
        gold = parse_gold(rec.get("ground_truth", ""))
        if ans is None:
            continue
        counts[ctype][1] += 1
        if ans == gold:
            counts[ctype][0] += 1
    return counts


def cell_accuracy(template, verb, model):
    """Per-cell accuracy as (acc_factual, acc_false). Uses n_total as denom
    (parse failures count as wrong — matches the existing Table 1)."""
    c = load_cell(template, verb, model)
    if c is None:
        return None, None
    def acc(ct):
        n_correct, n_parsed, n_total = c[ct]
        return n_correct / n_total if n_total else float("nan")
    return acc("factual"), acc("false")


def verb_template_avg(template, verb):
    """Average over the 10-model roster."""
    fa_list, fb_list = [], []
    for m in MODELS:
        af, ab = cell_accuracy(template, verb, m)
        if af is not None:
            fa_list.append(af)
        if ab is not None:
            fb_list.append(ab)
    if not fa_list or not fb_list:
        return None
    return (sum(fa_list) / len(fa_list),
            sum(fb_list) / len(fb_list),
            len(fa_list))


def avg_over_verbs(template, verbs):
    """Average over both models and the verbs in `verbs`."""
    fa_all, fb_all = [], []
    for v in verbs:
        for m in MODELS:
            af, ab = cell_accuracy(template, v, m)
            if af is not None:
                fa_all.append(af)
            if ab is not None:
                fb_all.append(ab)
    return (sum(fa_all) / len(fa_all),
            sum(fb_all) / len(fb_all),
            len(fa_all))


def fmt_row(template, fa, fb, n=None):
    gap = (fa - fb) * 100
    s = f"  {TEMPLATE_LABEL[template]:<24s}  factual={fa*100:5.1f}  false={fb*100:5.1f}  gap={gap:5.1f}"
    if n is not None:
        s += f"  (n_cells={n})"
    return s


def main():
    print("=" * 80)
    print("(A) Original 4-verb subset {believe, think, suppose, am_certain}")
    print("=" * 80)
    for tpl in TEMPLATES:
        fa, fb, n = avg_over_verbs(tpl, POSITIVE_4)
        print(fmt_row(tpl, fa, fb, n))

    print()
    print("=" * 80)
    print("(B) Per-verb breakdown — each new verb under all 4 templates")
    print("=" * 80)
    for verb in POSITIVE_4 + NEW_VERBS:
        print(f"\nverb = {verb}")
        for tpl in TEMPLATES:
            r = verb_template_avg(tpl, verb)
            if r is None:
                print(f"  {TEMPLATE_LABEL[tpl]:<24s}  [missing]")
            else:
                fa, fb, n = r
                print(fmt_row(tpl, fa, fb, n))

    print()
    print("=" * 80)
    print("(C) Per-family averages (averaged over verbs within family)")
    print("=" * 80)
    families = {
        "positive (4 verbs)": POSITIVE_4,
        "evidential: vaguely_remember": ["vaguely_remember"],
        "confidence: am_80_confident": ["am_80_confident"],
        "negation: seriously_doubt": ["seriously_doubt"],
    }
    for fam_name, verbs in families.items():
        print(f"\n{fam_name}")
        for tpl in TEMPLATES:
            fa, fb, n = avg_over_verbs(tpl, verbs)
            print(fmt_row(tpl, fa, fb, n))


if __name__ == "__main__":
    main()
