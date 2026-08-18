"""Concatenate two suppression result files into one, appending per-alpha item lists."""

import json
import os
import sys


def merge(base_path: str, incr_path: str, out_path: str) -> None:
    """Concatenate two result files' per-alpha item lists and write the merged JSON to out_path."""
    base = json.load(open(base_path))
    incr = json.load(open(incr_path))
    assert base["model"] == incr["model"], "model mismatch"
    assert base["alphas"] == incr["alphas"], "alphas mismatch"

    merged_results = []
    for br, ir in zip(base["results"], incr["results"]):
        assert br["alpha"] == ir["alpha"], f"alpha mismatch at {br['alpha']} vs {ir['alpha']}"
        merged_items = br["items"] + ir["items"]
        n_items = len(merged_items)
        n_parsed = sum(1 for it in merged_items if it.get("answer") is not None)
        n_correct = sum(1 for it in merged_items if it.get("answer") == it.get("gold"))
        acc = n_correct / n_items
        merged_results.append({
            "alpha": br["alpha"],
            "n_items": n_items,
            "n_parsed": n_parsed,
            "n_correct": n_correct,
            "acc": acc,
            "items": merged_items,
        })

    n_total = base["n_items"] + incr["n_items"]
    out = {"model": base["model"], "n_items": n_total,
           "alphas": base["alphas"], "results": merged_results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"merged: n_items {base['n_items']} + {incr['n_items']} = {n_total}")
    print(f"wrote {out_path}")
    for r in merged_results:
        print(f"  α={r['alpha']:>6}  acc={r['acc']:.3f}  ({r['n_correct']}/{r['n_items']})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: merge_suppress_results.py BASE INCREMENT OUT")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
