"""Aggregate results.jsonl into mAA / median error / runtime tables."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Tutorial's own default thresholds for calc_mAA (data/tutorial/metrics.py):
# 10 points log-spaced between 1 and 20 px, inclusive comparison (err <= th).
ACC_THRESHOLDS = np.logspace(np.log2(1.0), np.log2(20), 10, base=2.0)


def maa(errs):
    errs = np.asarray(errs)
    return float(np.mean([(errs <= t).mean() for t in ACC_THRESHOLDS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.jsonl")
    args = ap.parse_args()

    groups = defaultdict(list)
    for line in open(ROOT / args.results):
        r = json.loads(line)
        groups[(r["dataset"], r["method"], r["th"])].append(r)

    datasets = sorted({k[0] for k in groups})
    for ds in datasets:
        print(f"\n## {ds}\n")
        print("| method | th (px) | mAA | median err | mean time (s) |")
        print("|---|---|---|---|---|")
        best = {}
        for (d, m, th), recs in sorted(groups.items()):
            if d != ds:
                continue
            errs = [r["err"] for r in recs]
            row = (m, th, maa(errs), float(np.median(errs)),
                   float(np.mean([r["time"] for r in recs])))
            print("| {} | {} | {:.4f} | {:.2f} | {:.4f} |".format(*row))
            if m not in best or row[2] > best[m][2]:
                best[m] = row
        print(f"\nBest per method ({ds}):")
        print("| method | th (px) | mAA | median err | mean time (s) |")
        print("|---|---|---|---|---|")
        for m, row in sorted(best.items(), key=lambda kv: -kv[1][2]):
            print("| {} | {} | {:.4f} | {:.2f} | {:.4f} |".format(*row))


if __name__ == "__main__":
    main()
