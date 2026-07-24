"""Plot the F-matrix time-mAA curve from time_maa_f.jsonl (IMC-paper style)."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.plot_time_maa import INK, MUTED, PAGE, draw_panel
from bench.report_f import maa_f


def load_curves(path):
    groups = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        groups[(r["scene"], r["method"], r["budget"], r["th"])].append(r)
    per_budget = defaultdict(dict)
    for (sc, m, budget, _th), recs in groups.items():
        score = maa_f([r["err_deg"] for r in recs])
        t = float(np.mean([r["time"] for r in recs]))
        cur = per_budget[(sc, m, budget)]
        if not cur or score > cur["maa"]:
            cur.update(maa=score, time=t)
    curves = defaultdict(lambda: defaultdict(list))
    for (sc, m, _budget), v in per_budget.items():
        curves[sc][m].append((v["time"], v["maa"]))
    for sc in curves:
        for m in curves[sc]:
            curves[sc][m].sort()
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/time_maa_f.jsonl")
    ap.add_argument("--out", default="results/time_maa_f.png")
    args = ap.parse_args()

    curves = load_curves(ROOT / args.results)
    scenes = sorted(curves)
    fig, axes = plt.subplots(1, len(scenes), figsize=(7.5 * len(scenes), 5.2),
                             facecolor=PAGE, squeeze=False)
    for ax, sc in zip(axes[0], scenes):
        draw_panel(ax, curves[sc], f"PhotoTourism {sc} — fundamental matrix",
                   stagger={"dssac": None, "pydegensac": -4, "cv2-ransac": 8,
                            "dssac-pmin0.1": 10, "vibesac": -10})
        ax.set_xlabel("mean time per pair (s, log scale)", color=MUTED, fontsize=9)
    axes[0, 0].set_ylabel("pose mAA (1–10°)", color=MUTED, fontsize=9)
    axes[0, 0].set_ylim(0, 0.5)
    axes[0, 0].legend(loc="lower right", fontsize=8.5, frameon=False,
                      labelcolor="#52514e")
    fig.suptitle("F estimation: accuracy vs. compute budget",
                 color=INK, fontsize=12.5)
    fig.text(0.5, 0.008,
             "budget: max iterations 10–6400 (pydegensac, cv2, vibesac); percentile step dp 0.3–0.015 (DS-SAC)\n"
             "best inlier threshold per point · per-method tuned SNN (0.75–0.85)",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = ROOT / args.out
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out, "and", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
