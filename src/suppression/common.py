"""Shared loaders and alpha* selection for the attention-suppression analysis."""
import json
import os

import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results/attention_suppress_decode",
)


def load(model, task, item_type):
    """Load the held-out result JSON for one model, task, and item type.

    Args:
        model: model short name.
        task: "confirmation" or "verification".
        item_type: "factual" or "false".

    Returns:
        the parsed result dict.
    """
    path = os.path.join(RESULTS_DIR, "held_out", f"{model}_{task}_{item_type}.json")
    return json.load(open(path))


def correct_vec(d, gold, lo, hi):
    """Return a per-alpha dict of {0,1} correctness arrays for items lo:hi.

    Args:
        d: a loaded result dict, with d["results"] each carrying "alpha" and "items".
        gold: the correct answer letter, "A" or "B".
        lo, hi: the item-slice bounds.

    Returns:
        a dict mapping each alpha to an int8 0/1 correctness array over items[lo:hi].
    """
    out = {}
    for r in d["results"]:
        sub = r["items"][lo:hi]
        out[r["alpha"]] = np.array(
            [1 if it.get("answer") == gold else 0 for it in sub], dtype=np.int8
        )
    return out


def select_alpha_star(model, sel_lo=0, sel_hi=50):
    """Return the nonzero alpha maximizing mean confirmation accuracy on the selection set.

    Args:
        model: model short name.
        sel_lo, sel_hi: bounds of the selection slice.

    Returns:
        the nonzero alpha maximizing mean confirmation accuracy on that slice.
    """
    cf = correct_vec(load(model, "confirmation", "false"), "A", sel_lo, sel_hi)
    cfa = correct_vec(load(model, "confirmation", "factual"), "A", sel_lo, sel_hi)
    alphas = sorted(cf.keys())
    nonzero = [a for a in alphas if a != 0]
    avg = {a: 0.5 * (cf[a].mean() + cfa[a].mean()) for a in alphas}
    return max(nonzero, key=lambda a: avg[a])
