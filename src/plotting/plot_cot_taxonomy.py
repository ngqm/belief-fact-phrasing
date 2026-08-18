"""Charts for the CoT reasoning-strategy taxonomy.

Reads the classifier output (classify_cot_strategies.py) and writes:

  visualization/cot_taxonomy/
    strategy_overall.png              # bar: items per strategy
    strategy_by_verb.png              # heatmap: strategy share per verb
    strategy_by_model.png             # heatmap: strategy share per model
    strategy_accuracy_by_type.png     # accuracy within each strategy, factual vs false
    factcheck_vs_no_factcheck.png     # fact-checking CoTs vs the rest, by claim type
"""
import os
import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.taxonomy.classify_cot_strategies import FACTCHECK_CATEGORY

plt.style.use("fivethirtyeight")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["font.family"] = "serif"

CLASSIFICATIONS_PATH = "results/taxonomy/classifications.jsonl"
OUT_DIR = "visualization/cot_taxonomy"

VERB_ORDER = [
    "believe", "think", "suppose", "am_confident", "am_certain",
    "vaguely_remember", "was_told", "read_online",
    "am_0_confident", "am_20_confident", "am_40_confident",
    "am_60_confident", "am_80_confident", "am_100_confident",
    "dont_believe", "dont_think", "dont_suppose", "seriously_doubt",
]
MODEL_ORDER = [
    "gemma-3-4b", "gemma-3-12b", "gemma-3-27b",
    "llama-3.2-3b", "llama-3.1-8b", "llama-3.3-70b",
    "qwen3.5-9b", "qwen3.5-27b",
]


def load():
    rows = [json.loads(line) for line in open(CLASSIFICATIONS_PATH)]
    df = pd.DataFrame(rows)
    counts = Counter(df["category"])
    strategies = [s for s, _ in counts.most_common() if s != "other"]
    if "other" in counts:
        strategies.append("other")
    return df, strategies


def plot_overall(df, strategies, out):
    counts = df["category"].value_counts().reindex(strategies)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(strategies[::-1], counts.values[::-1], color="#3b6fb6")
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + 30, i, f"{v}  ({v/len(df)*100:.1f}%)",
                va="center", fontsize=10)
    ax.set_xlabel(f"CoTs (n={len(df)})")
    ax.set_title("Reasoning-strategy distribution")
    ax.set_xlim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _share_matrix(df, row_key, row_order, strategies):
    rows, cols = len(row_order), len(strategies)
    M = np.zeros((rows, cols))
    for i, r in enumerate(row_order):
        sub = df[df[row_key] == r]
        if len(sub) == 0:
            continue
        c = sub["category"].value_counts(normalize=True)
        for j, s in enumerate(strategies):
            M[i, j] = c.get(s, 0.0)
    return M


def _heatmap(M, row_labels, col_labels, title, out, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v*100:.0f}",
                        ha="center", va="center",
                        fontsize=8,
                        color="white" if v > 0.5 else "#222")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("share within row")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_by_verb(df, strategies, out):
    M = _share_matrix(df, "verb", VERB_ORDER, strategies)
    _heatmap(M, VERB_ORDER, strategies,
             "Strategy share by verb (% of CoTs in row)",
             out, figsize=(11, 7.5))


def plot_by_model(df, strategies, out):
    M = _share_matrix(df, "model", MODEL_ORDER, strategies)
    _heatmap(M, MODEL_ORDER, strategies,
             "Strategy share by model (% of CoTs in row)",
             out, figsize=(11, 5))


def plot_accuracy_by_type(df, strategies, out):
    types = ["factual", "false"]
    acc = np.full((len(strategies), len(types)), np.nan)
    n = np.zeros((len(strategies), len(types)), dtype=int)
    for i, s in enumerate(strategies):
        for j, t in enumerate(types):
            sub = df[(df["category"] == s) & (df["type"] == t)]
            if len(sub):
                acc[i, j] = sub["correct"].mean() * 100
                n[i, j] = len(sub)

    x = np.arange(len(strategies))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, acc[:, 0], w, label="factual", color="#3b6fb6")
    bars2 = ax.bar(x + w/2, acc[:, 1], w, label="false",   color="#d1495b")
    for i in range(len(strategies)):
        for k, bars in enumerate([bars1, bars2]):
            v = acc[i, k]
            if not np.isnan(v):
                ax.text(bars[i].get_x() + bars[i].get_width()/2,
                        v + 1.2, f"{v:.0f}\n(n={n[i,k]})",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Accuracy within each strategy, by claim type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_factcheck_vs_no_factcheck(df, out):
    df = df.copy()
    df["fc"] = df["category"] == FACTCHECK_CATEGORY

    types = ["factual", "false"]
    groups = [("fact-check", True), ("no fact-check", False)]

    fig, (ax_overall, ax_verb) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 2.4]}
    )

    # Overall panel
    x = np.arange(len(groups))
    w = 0.4
    for k, t in enumerate(types):
        ys, ns = [], []
        for _, fc in groups:
            sub = df[(df["fc"] == fc) & (df["type"] == t)]
            ys.append(sub["correct"].mean() * 100 if len(sub) else np.nan)
            ns.append(len(sub))
        bars = ax_overall.bar(x + (k - 0.5) * w, ys, w,
                              label=t,
                              color="#3b6fb6" if t == "factual" else "#d1495b")
        for i, (v, n) in enumerate(zip(ys, ns)):
            if not np.isnan(v):
                ax_overall.text(bars[i].get_x() + w / 2, v + 1.2,
                                f"{v:.0f}\n(n={n})",
                                ha="center", va="bottom", fontsize=9)
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels([g[0] for g in groups])
    ax_overall.set_ylim(0, 110)
    ax_overall.set_ylabel("Accuracy (%)")
    ax_overall.set_title(f"Overall (n={len(df)})")
    ax_overall.legend(loc="lower left")

    # Per-verb panel: factual vs false gap, split by whether the CoT fact-checks
    rows = []
    for v in VERB_ORDER:
        sub_v = df[df["verb"] == v]
        for label, fc in groups:
            for t in types:
                s = sub_v[(sub_v["fc"] == fc) & (sub_v["type"] == t)]
                rows.append({
                    "verb": v, "group": label, "type": t,
                    "acc": s["correct"].mean() * 100 if len(s) else np.nan,
                    "n": len(s),
                })
    pv = pd.DataFrame(rows)

    xv = np.arange(len(VERB_ORDER))
    w2 = 0.2
    offsets = {("fact-check", "factual"): -1.5,
               ("fact-check", "false"):   -0.5,
               ("no fact-check", "factual"): 0.5,
               ("no fact-check", "false"):   1.5}
    colors = {("fact-check", "factual"): "#3b6fb6",
              ("fact-check", "false"):   "#d1495b",
              ("no fact-check", "factual"): "#9bbde0",
              ("no fact-check", "false"):   "#e8a4ad"}
    for (g, t), off in offsets.items():
        sel = pv[(pv["group"] == g) & (pv["type"] == t)].set_index("verb").reindex(VERB_ORDER)
        ax_verb.bar(xv + off * w2, sel["acc"].values, w2,
                    label=f"{g} · {t}", color=colors[(g, t)])
    ax_verb.set_xticks(xv)
    ax_verb.set_xticklabels(VERB_ORDER, rotation=45, ha="right", fontsize=9)
    ax_verb.set_ylim(0, 110)
    ax_verb.set_ylabel("Accuracy (%)")
    ax_verb.set_title("By verb")
    ax_verb.legend(loc="lower left", fontsize=8, ncol=2)

    fig.suptitle("Performance: CoTs that fact-check vs CoTs that don't",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, strategies = load()
    print(f"loaded {len(df)} classified CoTs; strategies: {strategies}")

    plot_overall(df, strategies, f"{OUT_DIR}/strategy_overall.png")
    plot_by_verb(df, strategies, f"{OUT_DIR}/strategy_by_verb.png")
    plot_by_model(df, strategies, f"{OUT_DIR}/strategy_by_model.png")
    plot_accuracy_by_type(df, strategies, f"{OUT_DIR}/strategy_accuracy_by_type.png")
    plot_factcheck_vs_no_factcheck(df, f"{OUT_DIR}/factcheck_vs_no_factcheck.png")
    print(f"wrote 5 charts → {OUT_DIR}/")


if __name__ == "__main__":
    main()
