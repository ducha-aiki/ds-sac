"""Staged SNN tuning from the (th x snn) grid in results_snn.jsonl.

Stage 1: at each method's best inlier threshold from the unfiltered protocol
(snn = 1.0 rows), find the SNN ratio threshold with the best mAA.
Stage 2: with that SNN fixed, re-find the best inlier threshold.

Writes the tuned map to results/best_snn.json for time_maa.py --snn-json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.report import maa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results_snn.jsonl")
    ap.add_argument("--out", default="results/best_snn.json")
    args = ap.parse_args()

    cells = defaultdict(list)  # (ds, m, th, snn) -> errs
    for line in open(ROOT / args.results):
        r = json.loads(line)
        cells[(r["dataset"], r["method"], r["th"], r["snn"])].append(r["err"])
    score = {k: maa(v) for k, v in cells.items()}
    keys = sorted(score)
    datasets = sorted({k[0] for k in keys})
    methods = sorted({k[1] for k in keys})

    snn_map = {}
    print("| dataset | method | th@snn=1.0 | mAA@snn=1.0 | best snn | mAA stage1 "
          "| re-tuned th | mAA stage2 |")
    print("|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for m in methods:
            # Stage 0: best th on the as-shipped matches (snn = 1.0).
            th0 = max((k[2] for k in keys if k[:2] == (ds, m) and k[3] == 1.0),
                      key=lambda th: score[(ds, m, th, 1.0)])
            maa0 = score[(ds, m, th0, 1.0)]
            # Stage 1: best snn at that th (ties -> least filtering).
            snn1 = max((k[3] for k in keys if k[:3] == (ds, m, th0)),
                       key=lambda s: (score[(ds, m, th0, s)], s))
            maa1 = score[(ds, m, th0, snn1)]
            # Stage 2: re-tune th at that snn.
            th2 = max((k[2] for k in keys if k[:2] == (ds, m) and k[3] == snn1),
                      key=lambda th: score[(ds, m, th, snn1)])
            maa2 = score[(ds, m, th2, snn1)]
            snn_map[f"{ds}/{m}"] = snn1
            print(f"| {ds} | {m} | {th0} | {maa0:.4f} | {snn1} | {maa1:.4f} "
                  f"| {th2} | {maa2:.4f} |")

    out = ROOT / args.out
    json.dump(snn_map, open(out, "w"), indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
