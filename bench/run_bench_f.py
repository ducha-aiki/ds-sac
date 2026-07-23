"""Fundamental-matrix benchmark on a PhotoTourism validation scene.

Metric: max(rotation, translation) angular error of the pose recovered from
the predicted F (converted to E with GT intrinsics), per ransac-benchmark-2025.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "rb2025"))

from bench.f_data import SCENES, iter_pairs_f
from bench.f_methods import F_METHODS
from ransac_benchmark_imc.evaluation import eval_single_result

THRESHOLDS_F = (0.5, 0.75, 1.0, 2.0)
FAIL_ERR = np.pi  # eval_single_result's own no-model sentinel (radians)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(F_METHODS))
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--snn", type=float, default=0.85,
                    help="SNN ratio filter applied to all methods (1.0 = off); "
                         "unlike the homography archive, this data ships "
                         "unfiltered and raw inlier ratios are often below "
                         "DS-SAC's p_min floor")
    ap.add_argument("--limit", type=int, default=0, help="pairs per scene, 0=all")
    ap.add_argument("--out", default="results/results_f.jsonl")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for scene in args.scenes:
            pairs = iter_pairs_f(scene)
            if args.limit:
                pairs = (p for _, p in zip(range(args.limit), pairs))
            for pair in tqdm(list(pairs), desc=scene):
                keep = pair["scores"] <= args.snn
                pts1, pts2 = pair["pts1"][keep], pair["pts2"][keep]
                corr = np.hstack([pts1, pts2])
                for m in args.methods:
                    for th in THRESHOLDS_F:
                        t0 = time.perf_counter()
                        try:
                            F, mask = F_METHODS[m](pts1, pts2, th)
                        except Exception as exc:
                            print(f"{m} raised on {scene}/{pair['name']} "
                                  f"th={th}: {exc}", file=sys.stderr)
                            F, mask = None, None
                        dt = time.perf_counter() - t0
                        if F is None or mask is None or mask.sum() < 8:
                            err = FAIL_ERR
                        else:
                            err = float(eval_single_result(
                                pair["R1"], pair["R2"], pair["T1"], pair["T2"],
                                corr[mask], pair["K1"], pair["K2"], F_pred=F))
                        rec = {"scene": scene, "pair": pair["name"], "method": m,
                               "th": th, "snn": args.snn,
                               "err_deg": float(np.rad2deg(err)),
                               "time": dt,
                               "ninl": int(mask.sum()) if mask is not None else 0}
                        f.write(json.dumps(rec) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
