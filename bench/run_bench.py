"""Run all methods over EVD + HPatches at several inlier thresholds."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "tutorial"))

from bench.data import DATASETS, iter_pairs
from bench.methods import METHODS
from metrics import get_visible_part_mean_absolute_reprojection_error as mae_err

THRESHOLDS = (0.5, 0.75, 1.0, 2.0, 4.0)
FAIL_ERR = 1e6  # sentinel when a method returns no model


def snn_filter(pair, snn_th):
    """Matches passing the SNN ratio test (scores are lower-is-better ratios).
    snn_th >= 1.0 keeps everything."""
    if snn_th >= 1.0 or pair["scores"] is None:
        return pair["pts1"], pair["pts2"]
    keep = pair["scores"] <= snn_th
    return pair["pts1"][keep], pair["pts2"][keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--snn", nargs="*", type=float, default=[1.0],
                    help="SNN ratio thresholds to sweep (1.0 = no filtering)")
    ap.add_argument("--limit", type=int, default=0, help="pairs per dataset, 0=all")
    ap.add_argument("--out", default="results/results.jsonl")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for ds in args.datasets:
            pairs = list(iter_pairs(ds))
            if args.limit:
                pairs = pairs[: args.limit]
            for pair in tqdm(pairs, desc=ds):
                for snn in args.snn:
                    pts1, pts2 = snn_filter(pair, snn)
                    for m in args.methods:
                        for th in THRESHOLDS:
                            t0 = time.perf_counter()
                            try:
                                H, mask = METHODS[m](pts1, pts2, th)
                            except Exception as exc:
                                print(f"{m} raised on {ds}/{pair['name']} "
                                      f"th={th} snn={snn}: {exc}", file=sys.stderr)
                                H, mask = None, None
                            dt = time.perf_counter() - t0
                            if H is None:
                                err = FAIL_ERR
                            else:
                                err = float(mae_err(pair["img1"], pair["img2"],
                                                    pair["H_gt"], H))
                            rec = {"dataset": ds, "pair": pair["name"], "method": m,
                                   "th": th, "snn": snn, "err": err, "time": dt,
                                   "ninl": int(np.sum(mask)) if mask is not None else 0}
                            f.write(json.dumps(rec) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
