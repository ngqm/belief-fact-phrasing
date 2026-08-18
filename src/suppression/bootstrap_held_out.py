"""Bootstrap standard errors for the attention-suppression results.

Confirm and Verify average accuracy on the held-out factual and false items
(KaBLE Task 5 and Task 4). Standard error is the std over B bootstrap resamples.
"""
import sys

import numpy as np

from src.suppression.common import load, correct_vec, select_alpha_star

REP_LO, REP_HI = 50, 500
B = 10000
RNG_SEED = 0


def bootstrap_se(values_a, values_b, B, rng):
    """Return the bootstrap SE of the mean of concat(values_a, values_b).

    Resamples each subset independently at its own length, then pools.

    Args:
        values_a, values_b: per-item 0/1 arrays for the two subsets.
        B: number of bootstrap resamples.
        rng: a numpy Generator.

    Returns:
        the bootstrap SE (float) of the pooled mean.
    """
    n_a = len(values_a)
    n_b = len(values_b)
    means = np.empty(B)
    for k in range(B):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        means[k] = (values_a[idx_a].sum() + values_b[idx_b].sum()) / (n_a + n_b)
    return float(means.std(ddof=1))


def run(model):
    """Print held-out Confirm and Verify accuracy at alpha=0 and alpha* for one model."""
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
