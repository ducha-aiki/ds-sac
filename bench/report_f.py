"""Aggregate results_f.jsonl into pose-error mAA tables (1-10 degrees)."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACC_THRESHOLDS_DEG = np.linspace(1.0, 10.0, 10)


def maa_f(errs_deg):
    errs = np.asarray(errs_deg)
    return float(np.mean([(errs <= t).mean() for t in ACC_THRESHOLDS_DEG]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results_f.jsonl")
    args = ap.parse_args()

    groups = defaultdict(list)
    for line in open(ROOT / args.results):
        r = json.loads(line)
        groups[(r["scene"], r["method"], r["th"], r.get("snn", 1.0))].append(r)

    show_snn = len({k[3] for k in groups}) > 1
    header = ("| method | th (px) | snn | mAA@10deg | median err (deg) | mean time (s) |"
              if show_snn else
              "| method | th (px) | mAA@10deg | median err (deg) | mean time (s) |")
    sep = "|---|" + "---|" * (5 if show_snn else 4)

    scenes = sorted({k[0] for k in groups})
    for sc in scenes:
        print(f"\n## {sc}\n")
        print(header)
        print(sep)
        best = {}
        for (s, m, th, snn), recs in sorted(groups.items()):
            if s != sc:
                continue
            errs = [r["err_deg"] for r in recs]
            stats = (maa_f(errs), float(np.median(errs)),
                     float(np.mean([r["time"] for r in recs])))
            row = (m, th) + ((snn,) if show_snn else ()) + stats
            print(("| " + " | ".join(["{}"] * (len(row) - 3))
                   + " | {:.4f} | {:.2f} | {:.4f} |").format(*row))
            if m not in best or stats[0] > best[m][1][0]:
                best[m] = (row[:-3], stats)
        print(f"\nBest per method ({sc}):")
        print(header)
        print(sep)
        for m, (key, stats) in sorted(best.items(), key=lambda kv: -kv[1][1][0]):
            row = key + stats
            print(("| " + " | ".join(["{}"] * len(key))
                   + " | {:.4f} | {:.2f} | {:.4f} |").format(*row))


if __name__ == "__main__":
    main()
