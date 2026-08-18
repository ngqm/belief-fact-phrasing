"""Paper figures.

Outputs under paper/figures/:
  fig1_verb_gap.pdf        per-verb factual/false accuracy bars (avg over models)
  fig2_confidence.pdf      accuracy vs stated confidence level
  fig3_strategies.pdf      strategy distribution + accuracy by claim type
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Anthropic-inspired palette: terracotta/clay primary, slate, muted sage.
ANTH = {
    "terra":      "#CC785C",   # primary terracotta
    "terra_dark": "#A8533B",
    "terra_lite": "#E5A584",
    "slate":      "#3E4A52",   # dark slate/charcoal
    "slate_lite": "#7A8A93",
    "sage":       "#4A6B5C",   # dark muted sage
    "sage_lite":  "#8FA985",
    "grey":       "#6B655E",   # warm neutral grey
}

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

VERB_DISPLAY = {
    "believe": "believe", "think": "think", "suppose": "suppose",
    "am_confident": "am confident", "am_certain": "am certain",
    "vaguely_remember": "vaguely remember", "was_told": "was told",
    "read_online": "read online",
    "am_0_confident": "am 0\\% confident", "am_20_confident": "am 20\\% confident",
    "am_40_confident": "am 40\\% confident", "am_60_confident": "am 60\\% confident",
    "am_80_confident": "am 80\\% confident", "am_100_confident": "am 100\\% confident",
    "dont_believe": "don't believe", "dont_think": "don't think",
    "dont_suppose": "don't suppose", "seriously_doubt": "seriously doubt",
}


def fig1_verb_gap():
    # Drop the am_X_confident verbs (covered in fig2); keep the remaining 12.
    keep = {
        "believe", "think", "suppose",
        "am_confident", "am_certain",
        "vaguely_remember", "was_told", "read_online",
        "dont_believe", "dont_think", "dont_suppose", "seriously_doubt",
    }
    df = pd.read_csv("analysis/original/master_accuracy_summary.csv")
    df = df[df["Model"] != "deepseek-v4-flash"]  # judge, not one of the 10 evaluated models
    df = df[df["Verb"].isin(keep)]
    agg = df.groupby(["Verb", "Type"])["Accuracy"].mean().unstack()
    agg = agg.sort_values("false", ascending=False)

    labels = [VERB_DISPLAY.get(v, v).replace("\\", "") for v in agg.index]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    for i, v in enumerate(agg.index):
        val_f  = agg.loc[v, "factual"]
        val_fa = agg.loc[v, "false"]
        ax.plot([val_f, val_fa], [i, i], color="black", lw=1.1,
                solid_capstyle="butt", zorder=2)
        ax.plot(val_f,  i, marker="o", color=ANTH["slate"],
                markeredgecolor="black", markeredgewidth=0.6, markersize=6, zorder=3)
        ax.plot(val_fa, i, marker="o", color=ANTH["terra"],
                markeredgecolor="black", markeredgewidth=0.6, markersize=6, zorder=3)
        if val_f >= val_fa:
            off_f, off_fa = (10, 0), (-10, 0)
        else:
            off_f, off_fa = (-10, 0), (10, 0)
        ax.annotate(f"{val_f:.0f}",  (val_f,  i), textcoords="offset points",
                    xytext=off_f,  ha="center", va="center", fontsize=6.5, zorder=4)
        ax.annotate(f"{val_fa:.0f}", (val_fa, i), textcoords="offset points",
                    xytext=off_fa, ha="center", va="center", fontsize=6.5, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xticks([])

    # Custom legend with proxy markers.
    ax.scatter([], [], marker="o", color=ANTH["slate"],
               edgecolor="black", linewidth=0.6, s=36, label="factual")
    ax.scatter([], [], marker="o", color=ANTH["terra"],
               edgecolor="black", linewidth=0.6, s=36, label="false")
    ax.legend(loc="lower left", frameon=True, fontsize=7.5,
              handletextpad=0.3)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color="#dddddd", lw=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_verb_gap.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_confidence():
    df = pd.read_csv("analysis/original/master_accuracy_summary.csv")
    verbs = ["am_0_confident", "am_20_confident", "am_40_confident",
             "am_60_confident", "am_80_confident", "am_100_confident"]
    levels = [0, 20, 40, 60, 80, 100]
    sub = df[df["Verb"].isin(verbs)]
    models = ["gemma-3-12b", "gemma-3-27b",
              "llama-3.3-70b",
              "qwen3.5-27b", "qwen3.5-35b-a3b"]  # five largest; judge excluded
    pretty = {
        "gemma-3-12b":       "Gemma 3 12B",
        "gemma-3-27b":       "Gemma 3 27B",
        "llama-3.3-70b":     "Llama 3.3 70B",
        "qwen3.5-27b":       "Qwen 3.5 27B",
        "qwen3.5-35b-a3b":   "Qwen 3.5 35B-A3B",
        "deepseek-v4-flash": "DeepSeek V4 Flash",
    }
    colors = {
        "gemma-3-12b":       ANTH["terra_lite"],
        "gemma-3-27b":       ANTH["terra_dark"],
        "llama-3.3-70b":     "#6B8EB5",   # light steel blue
        "qwen3.5-27b":       "#2F6B40",   # forest green
        "qwen3.5-35b-a3b":   "#7DB585",   # lighter green
        "deepseek-v4-flash": "#6E4F99",   # muted purple
    }

    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.6), sharex=True)
    for ax, t, title in [(axes[0], "factual", "(a) factual claims"),
                         (axes[1], "false", "(b) false claims")]:
        for m in models:
            row = []
            for v in verbs:
                r = sub[(sub.Model == m) & (sub.Type == t) & (sub.Verb == v)].Accuracy.values
                row.append(r[0] if len(r) else np.nan)
            ax.plot(levels, row, marker="o", color=colors[m], label=pretty[m], lw=1.2, markersize=3.5)
        ax.set_xticks(levels)
        ax.set_ylim(20, 105)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("Accuracy (%)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[1].set_xlabel("Stated confidence (%)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=6.5,
                   ncol=2, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_confidence.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_strategies():
    rows = [json.loads(line) for line in open("results/taxonomy/classifications.jsonl")]
    df = pd.DataFrame(rows)
    df = df[df["model"] != "deepseek-v4-flash"]  # judge, not one of the 10 evaluated models

    strategies = [s for s in ["factual_verification", "logical_affirmation",
                              "direct_repetition", "no_reasoning",
                              "subjectivity_deflection"]
                  if s in df["category"].unique()]
    display = {
        "factual_verification": "factual verif.",
        "logical_affirmation": "logical affirm.",
        "direct_repetition": "direct repet.",
        "no_reasoning": "no reasoning",
        "subjectivity_deflection": "subj. deflect.",
    }

    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.6),
                             gridspec_kw={"height_ratios": [1, 1]})
    counts = df["category"].value_counts().reindex(strategies)
    axes[0].bar(range(len(strategies)), counts.values, width=0.5, color=ANTH["grey"])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 60, f"{v/len(df)*100:.0f}%",
                     ha="center", fontsize=7.5)
    axes[0].set_xticks(range(len(strategies)))
    axes[0].set_xticklabels([display[s] for s in strategies], fontsize=7,
                            rotation=25, ha="right")
    axes[0].set_title("(a) Strategy distribution", fontsize=8.5)
    axes[0].set_ylim(0, counts.max() * 1.22)
    axes[0].set_yticks([])
    for spine in ("top", "right", "left"):
        axes[0].spines[spine].set_visible(False)

    # Right subplot: fact-checking vs non-fact-checking accuracy.
    df["group"] = np.where(df["category"] == "factual_verification",
                           "fact-checking", "non-fact-checking")
    groups = ["fact-checking", "non-fact-checking"]
    acc = np.full((len(groups), 2), np.nan)
    for i, g in enumerate(groups):
        for j, t in enumerate(["factual", "false"]):
            sub = df[(df["group"] == g) & (df["type"] == t)]
            if len(sub):
                acc[i, j] = sub["correct"].mean() * 100
    x = np.arange(len(groups))
    w = 0.22
    axes[1].bar(x - w / 2, acc[:, 0], w, color=ANTH["slate"], label="factual")
    axes[1].bar(x + w / 2, acc[:, 1], w, color=ANTH["terra"], label="false",
                hatch="///", edgecolor="white", linewidth=0)
    for i in range(len(groups)):
        for j, dx in enumerate([-w / 2, w / 2]):
            v = acc[i, j]
            if not np.isnan(v):
                axes[1].text(i + dx, v + 1.5, f"{v:.0f}",
                             ha="center", fontsize=7.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, fontsize=8)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 115)
    axes[1].legend(loc="upper left", frameon=False, fontsize=7.5)
    axes[1].set_title("(b) Fact-checking vs not")
    for spine in ("top", "right"):
        axes[1].spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_strategies.pdf")
    plt.close(fig)


def main():
    fig1_verb_gap()
    fig2_confidence()
    fig3_strategies()
    print(f"wrote 3 figures -> {OUT}/")


if __name__ == "__main__":
    main()
