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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
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
                for m in args.methods:
                    for th in THRESHOLDS:
                        t0 = time.perf_counter()
                        H, mask = METHODS[m](pair["pts1"], pair["pts2"], th)
                        dt = time.perf_counter() - t0
                        if H is None:
                            err = FAIL_ERR
                        else:
                            err = float(mae_err(pair["img1"], pair["img2"],
                                                pair["H_gt"], H))
                        rec = {"dataset": ds, "pair": pair["name"], "method": m,
                               "th": th, "err": err, "time": dt,
                               "ninl": int(np.sum(mask)) if mask is not None else 0}
                        f.write(json.dumps(rec) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
