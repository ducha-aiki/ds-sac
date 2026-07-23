"""Plot the time-mAA curve (IMC-paper style) from time_maa.jsonl.

Each point is one compute budget: x = measured mean runtime per pair, y = the
method's best mAA over the inlier-threshold sweep at that budget.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.report import maa

# Categorical palette (validated, fixed assignment — color follows the method).
COLORS = {
    "dssac": "#2a78d6",
    "pydegensac": "#eb6834",
    "cv2-ransac": "#1baf7a",
    "cv2-magsac": "#eda100",
}
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def load_curves(path):
    """-> {dataset: {method: [(mean_time, best_maa), ...] sorted by time}}"""
    groups = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        groups[(r["dataset"], r["method"], r["budget"], r["th"])].append(r)

    per_budget = defaultdict(dict)  # (ds, m, budget) -> best mAA / mean time
    for (ds, m, budget, _th), recs in groups.items():
        score = maa([r["err"] for r in recs])
        t = float(np.mean([r["time"] for r in recs]))
        cur = per_budget[(ds, m, budget)]
        if not cur or score > cur["maa"]:
            cur.update(maa=score, time=t)

    curves = defaultdict(lambda: defaultdict(list))
    for (ds, m, _budget), v in per_budget.items():
        curves[ds][m].append((v["time"], v["maa"]))
    for ds in curves:
        for m in curves[ds]:
            curves[ds][m].sort()
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/time_maa.jsonl")
    ap.add_argument("--out", default="results/time_maa.png")
    args = ap.parse_args()

    curves = load_curves(ROOT / args.results)
    datasets = [ds for ds in ("EVD", "HPatchesSeq") if ds in curves] or sorted(curves)

    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 4.2), sharey=True,
                             facecolor=PAGE)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        for m, color in COLORS.items():
            if m not in curves[ds]:
                continue
            pts = np.array(curves[ds][m])
            ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, linewidth=2,
                    markersize=7, label=m, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(m, (pts[-1, 0], pts[-1, 1]), textcoords="offset points",
                        xytext=(6, 5), fontsize=8.5, color=INK2)
        ax.set_title(ds, color=INK, fontsize=11)
        ax.set_xlabel("mean time per pair (s, log scale)", color=MUTED, fontsize=9)
        ax.grid(True, which="major", color=GRID, linewidth=0.75, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.margins(x=0.18)
    axes[0].set_ylabel("mAA (1–20 px, log-spaced)", color=MUTED, fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=False,
                   labelcolor=INK2)
    fig.suptitle("Homography estimation: accuracy vs. compute budget",
                 color=INK, fontsize=12.5)
    fig.text(0.5, 0.008,
             "budget: max iterations 10–6400 (pydegensac, cv2); percentile step "
             "dp 0.3–0.015 (DS-SAC) · best inlier threshold per point · "
             "unfiltered tentative matches",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out, "and", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
