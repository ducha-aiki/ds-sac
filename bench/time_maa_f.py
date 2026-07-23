"""Budget sweep for the F-matrix time-mAA curve, at tuned per-method SNN."""
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
from bench.f_methods import F_BUDGETS
from bench.run_bench_f import FAIL_ERR, THRESHOLDS_F
from ransac_benchmark_imc.evaluation import eval_single_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(F_BUDGETS))
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--limit", type=int, default=0, help="pairs per scene, 0=all")
    ap.add_argument("--out", default="results/time_maa_f.jsonl")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for scene in args.scenes:
            pairs = list(iter_pairs_f(scene))
            if args.limit:
                pairs = pairs[: args.limit]
            for m in args.methods:
                budgets, factory, snn = F_BUDGETS[m]
                for budget in tqdm(budgets, desc=f"{scene}/{m}"):
                    fn = factory(budget)
                    for pair in pairs:
                        keep = pair["scores"] <= snn
                        pts1, pts2 = pair["pts1"][keep], pair["pts2"][keep]
                        corr = np.hstack([pts1, pts2])
                        for th in THRESHOLDS_F:
                            t0 = time.perf_counter()
                            try:
                                F, mask = fn(pts1, pts2, th)
                            except Exception as exc:
                                print(f"{m} raised on {scene}/{pair['name']} "
                                      f"budget={budget} th={th}: {exc}",
                                      file=sys.stderr)
                                F, mask = None, None
                            dt = time.perf_counter() - t0
                            if F is None or mask is None or mask.sum() < 8:
                                err = FAIL_ERR
                            else:
                                err = float(eval_single_result(
                                    pair["R1"], pair["R2"], pair["T1"],
                                    pair["T2"], corr[mask], pair["K1"],
                                    pair["K2"], F_pred=F))
                            rec = {"scene": scene, "pair": pair["name"],
                                   "method": m, "budget": budget, "snn": snn,
                                   "th": th, "err_deg": float(np.rad2deg(err)),
                                   "time": dt}
                            f.write(json.dumps(rec) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
