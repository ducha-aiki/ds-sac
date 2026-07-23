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
        groups[(r["dataset"], r["method"], r["th"], r.get("snn", 1.0))].append(r)

    show_snn = len({k[3] for k in groups}) > 1
    header = ("| method | th (px) | snn | mAA | median err | mean time (s) |"
              if show_snn else
              "| method | th (px) | mAA | median err | mean time (s) |")
    sep = "|---|" + "---|" * (5 if show_snn else 4)

    datasets = sorted({k[0] for k in groups})
    for ds in datasets:
        print(f"\n## {ds}\n")
        print(header)
        print(sep)
        best = {}
        for (d, m, th, snn), recs in sorted(groups.items()):
            if d != ds:
                continue
            errs = [r["err"] for r in recs]
            # mAA counts failures (err = 1e6 sentinel) as misses; the median is
            # reported over successful estimates only so a >50% failure rate
            # shows up as a high mAA loss, not a nonsensical 1e6 median.
            ok = [e for e in errs if e < 1e6] or [float("inf")]
            stats = (maa(errs), float(np.median(ok)),
                     float(np.mean([r["time"] for r in recs])))
            row = (m, th) + ((snn,) if show_snn else ()) + stats
            print(("| " + " | ".join(["{}"] * (len(row) - 3))
                   + " | {:.4f} | {:.2f} | {:.4f} |").format(*row))
            if m not in best or stats[0] > best[m][1][0]:
                best[m] = (row[:-3], stats)
        print(f"\nBest per method ({ds}):")
        print(header)
        print(sep)
        for m, (key, stats) in sorted(best.items(), key=lambda kv: -kv[1][1][0]):
            row = key + stats
            print(("| " + " | ".join(["{}"] * len(key))
                   + " | {:.4f} | {:.2f} | {:.4f} |").format(*row))


if __name__ == "__main__":
    main()
