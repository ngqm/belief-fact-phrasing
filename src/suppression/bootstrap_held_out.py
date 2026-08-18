r"""Bootstrap standard errors on the held-out report subset for Tables 2 and 3.

For each (model, alpha) pair, the Confirm column averages accuracy on
100 factual + 100 false items (KABLE Task 5); the Verify column averages
accuracy on 100 factual + 100 false items (KABLE Task 4). We bootstrap
across the 200-item combined held-out set per column, resampling with
replacement B times, and report SE = std of the bootstrap distribution.

Output: per-model lines suitable for pasting subscript-notation values
($x_{\pm s}$) into the paper.
"""
import json
import os
import sys
import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results/attention_suppress_decode",
)

REP_LO, REP_HI = 50, 150
B = 10000
RNG_SEED = 0


def load(model, task, item_type):
    path = os.path.join(RESULTS_DIR, f"{model}_n150_{task}_{item_type}.json")
    return json.load(open(path))


def correct_vec(d, gold, lo, hi):
    """Per-alpha dict of length-(hi-lo) {0,1} arrays."""
    out = {}
    for r in d["results"]:
        sub = r["items"][lo:hi]
        out[r["alpha"]] = np.array(
            [1 if it.get("answer") == gold else 0 for it in sub], dtype=np.int8
        )
    return out


def bootstrap_se(values_a, values_b, B, rng):
    """SE of the mean of concat(values_a, values_b) via bootstrap.
    Resamples each subset independently (preserving 100+100 structure)."""
    n_a = len(values_a)
    n_b = len(values_b)
    means = np.empty(B)
    for k in range(B):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        means[k] = (values_a[idx_a].sum() + values_b[idx_b].sum()) / (n_a + n_b)
    return float(means.std(ddof=1))


def select_alpha_star(model):
    """Replicate the held-out alpha* selection used in the paper."""
    cf_sel = correct_vec(load(model, "confirmation", "false"), "A", 0, 50)
    cfa_sel = correct_vec(load(model, "confirmation", "factual"), "A", 0, 50)
    alphas = sorted(cf_sel.keys())
    nonzero = [a for a in alphas if a != 0]
    avg_sel = {a: 0.5 * (cf_sel[a].mean() + cfa_sel[a].mean()) for a in alphas}
    return max(nonzero, key=lambda a: avg_sel[a])


def run(model):
    rng = np.random.default_rng(RNG_SEED)
    alpha_star = select_alpha_star(model)

    cf = correct_vec(load(model, "confirmation", "false"), "A", REP_LO, REP_HI)
    cfa = correct_vec(load(model, "confirmation", "factual"), "A", REP_LO, REP_HI)
    vf = correct_vec(load(model, "verification", "false"), "B", REP_LO, REP_HI)
    vfa = correct_vec(load(model, "verification", "factual"), "A", REP_LO, REP_HI)

    print(f"\n=== {model}  (alpha* = {alpha_star}) ===")
    for label, alpha in [("alpha=0", 0.0), (f"alpha*={alpha_star}", alpha_star)]:
        c_acc = 0.5 * (cf[alpha].mean() + cfa[alpha].mean()) * 100
        c_se = bootstrap_se(cf[alpha], cfa[alpha], B, rng) * 100
        v_acc = 0.5 * (vf[alpha].mean() + vfa[alpha].mean()) * 100
        v_se = bootstrap_se(vf[alpha], vfa[alpha], B, rng) * 100
        print(f"  {label:>14}  Confirm = {c_acc:5.1f} +/- {c_se:.1f}   "
              f"Verify = {v_acc:5.1f} +/- {v_se:.1f}")


if __name__ == "__main__":
    models = sys.argv[1:] or [
        "llama-3.1-8b", "qwen-3.5-9b",
        "gemma-3-12b", "qwen-3.5-4b", "gemma-3-4b", "llama-3.2-3b",
    ]
    for m in models:
        run(m)
