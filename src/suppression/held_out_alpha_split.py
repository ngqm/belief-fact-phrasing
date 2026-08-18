"""Held-out alpha* selection: pick alpha on items 0:50, report on items 50:150.

For each model:
  - For each alpha in the n=150 run, split the per-item list into selection
    (first 50) and report (next 100) for confirmation-{false,factual} and
    verification-{false,factual}.
  - alpha* := argmax over alpha != 0 of mean(confirm-false-sel, confirm-factual-sel).
  - Print Confirm and Verify accuracies on the report subset at alpha=0 and alpha*.
"""

import json
import os
import sys


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results/attention_suppress_decode",
)


def load(model, task, item_type):
    path = os.path.join(RESULTS_DIR, f"{model}_n150_{task}_{item_type}.json")
    return json.load(open(path))


def split_acc(items, gold, lo, hi):
    sub = items[lo:hi]
    n_correct = sum(1 for it in sub if it.get("answer") == gold)
    return n_correct / len(sub) if sub else 0.0


def per_alpha(d, gold, lo, hi):
    return {r["alpha"]: split_acc(r["items"], gold, lo, hi) for r in d["results"]}


def main(model):
    cf = load(model, "confirmation", "false")        # gold A
    cfa = load(model, "confirmation", "factual")     # gold A
    vf = load(model, "verification", "false")        # gold B
    vfa = load(model, "verification", "factual")     # gold A

    SEL_LO, SEL_HI = 0, 50
    REP_LO, REP_HI = 50, 150

    cf_sel = per_alpha(cf, "A", SEL_LO, SEL_HI)
    cfa_sel = per_alpha(cfa, "A", SEL_LO, SEL_HI)
    cf_rep = per_alpha(cf, "A", REP_LO, REP_HI)
    cfa_rep = per_alpha(cfa, "A", REP_LO, REP_HI)
    vf_rep = per_alpha(vf, "B", REP_LO, REP_HI)
    vfa_rep = per_alpha(vfa, "A", REP_LO, REP_HI)

    alphas = sorted(cf_sel.keys())
    nonzero = [a for a in alphas if a != 0]
    confirm_avg_sel = {a: 0.5 * (cf_sel[a] + cfa_sel[a]) for a in alphas}
    alpha_star = max(nonzero, key=lambda a: confirm_avg_sel[a])

    def confirm_avg_rep(a):
        return 0.5 * (cf_rep[a] + cfa_rep[a])

    def verify_avg_rep(a):
        return 0.5 * (vf_rep[a] + vfa_rep[a])

    print(f"=== {model} ===")
    print(f"selection set: items[{SEL_LO}:{SEL_HI}] (n=50 per claim-type)")
    print(f"report set:    items[{REP_LO}:{REP_HI}] (n=100 per claim-type)")
    print()
    print(f"{'alpha':>6}  {'sel-cf':>7}  {'sel-cfa':>7}  {'sel-cavg':>8}  "
          f"{'rep-cf':>7}  {'rep-cfa':>7}  {'rep-cavg':>8}  "
          f"{'rep-vf':>7}  {'rep-vfa':>7}  {'rep-vavg':>8}")
    for a in alphas:
        marker = "  <-- alpha*" if a == alpha_star else ""
        print(f"{a:>6}  {cf_sel[a]:>7.3f}  {cfa_sel[a]:>7.3f}  "
              f"{confirm_avg_sel[a]:>8.3f}  "
              f"{cf_rep[a]:>7.3f}  {cfa_rep[a]:>7.3f}  "
              f"{confirm_avg_rep(a):>8.3f}  "
              f"{vf_rep[a]:>7.3f}  {vfa_rep[a]:>7.3f}  "
              f"{verify_avg_rep(a):>8.3f}{marker}")
    print()
    print(f"=> alpha* = {alpha_star}")
    print(f"   Confirm @ a=0 : {confirm_avg_rep(0.0)*100:.1f}%")
    print(f"   Confirm @ a* : {confirm_avg_rep(alpha_star)*100:.1f}%")
    print(f"   delta Confirm : {(confirm_avg_rep(alpha_star)-confirm_avg_rep(0.0))*100:+.1f}%")
    print(f"   Verify  @ a=0 : {verify_avg_rep(0.0)*100:.1f}%")
    print(f"   Verify  @ a* : {verify_avg_rep(alpha_star)*100:.1f}%")
    print(f"   delta Verify  : {(verify_avg_rep(alpha_star)-verify_avg_rep(0.0))*100:+.1f}%")
    print()


if __name__ == "__main__":
    models = sys.argv[1:] or [
        "llama-3.1-8b", "qwen-3.5-9b",
        "gemma-3-12b", "gemma-3-4b", "qwen-3.5-4b", "llama-3.2-3b",
    ]
    for m in models:
        main(m)
