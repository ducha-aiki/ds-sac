"""Sweep compute budgets to produce data for the time-mAA curve (IMC-style)."""
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
from bench.methods import BUDGETS
from bench.run_bench import FAIL_ERR, THRESHOLDS, snn_filter
from metrics import get_visible_part_mean_absolute_reprojection_error as mae_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(BUDGETS))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--snn-json", default=None,
                    help='JSON file mapping "dataset/method" -> tuned SNN threshold')
    ap.add_argument("--limit", type=int, default=0, help="pairs per dataset, 0=all")
    ap.add_argument("--out", default="results/time_maa.jsonl")
    args = ap.parse_args()

    snn_map = json.load(open(ROOT / args.snn_json)) if args.snn_json else {}

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for ds in args.datasets:
            pairs = list(iter_pairs(ds))
            if args.limit:
                pairs = pairs[: args.limit]
            for m in args.methods:
                snn = snn_map.get(f"{ds}/{m}", 1.0)
                budgets, factory = BUDGETS[m]
                for budget in tqdm(budgets, desc=f"{ds}/{m}"):
                    fn = factory(budget)
                    for pair in pairs:
                        pts1, pts2 = snn_filter(pair, snn)
                        for th in THRESHOLDS:
                            t0 = time.perf_counter()
                            try:
                                H, mask = fn(pts1, pts2, th)
                            except Exception as exc:
                                print(f"{m} raised on {ds}/{pair['name']} "
                                      f"budget={budget} th={th}: {exc}",
                                      file=sys.stderr)
                                H = None
                            dt = time.perf_counter() - t0
                            if H is None:
                                err = FAIL_ERR
                            else:
                                err = float(mae_err(pair["img1"], pair["img2"],
                                                    pair["H_gt"], H))
                            rec = {"dataset": ds, "pair": pair["name"], "method": m,
                                   "budget": budget, "th": th, "snn": snn,
                                   "err": err, "time": dt}
                            f.write(json.dumps(rec) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
