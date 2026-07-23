"""Speed + equivalence harness for optimizing dssac.

--save writes golden H matrices for a fixed pair subset; --check compares the
current implementation against the golden file (exact or corner-error bound)
and reports timing. DS-SAC is deterministic, so pure refactors must match
exactly; fp-reordering changes must stay within --tol px corner error.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dssac
from bench.data import iter_pairs

N_HP, N_EVD = 25, 7
THRESHOLD = 2.0


def corner_err(Ha, Hb, w=1000, h=1000):
    c = np.array([[0., 0., 1.], [w, 0., 1.], [w, h, 1.], [0., h, 1.]])
    pa = c @ Ha.T
    pb = c @ Hb.T
    return np.abs(pa[:, :2] / pa[:, 2:] - pb[:, :2] / pb[:, 2:]).max()


def run(backend):
    out, times = {}, []
    for ds, n in (("HPatchesSeq", N_HP), ("EVD", N_EVD)):
        for pair in list(iter_pairs(ds))[:n]:
            t0 = time.perf_counter()
            H, _ = dssac.find_homography(pair["pts1"], pair["pts2"], THRESHOLD,
                                         backend=backend)
            times.append(time.perf_counter() - t0)
            out[f"{ds}/{pair['name']}"] = H if H is not None else np.full((3, 3), np.nan)
    return out, float(np.mean(times)), float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="allowed max corner error vs golden (0 = exact match)")
    ap.add_argument("--backend", default="auto",
                    help="dssac backend; goldens are only comparable per backend")
    ap.add_argument("--golden", default="results/golden.npz")
    args = ap.parse_args()

    if args.backend == "numba":
        run(args.backend)  # warm up JIT outside the timed pass
    out, mean_t, med_t = run(args.backend)
    print(f"{len(out)} pairs: mean {mean_t*1000:.2f} ms, median {med_t*1000:.2f} ms")

    path = ROOT / args.golden
    if args.save:
        np.savez(path, **out)
        print("saved golden to", path)
        return

    golden = np.load(path)
    worst = ("", -1.0)
    for k, H in out.items():
        G = golden[k]
        if np.isnan(G).any() != np.isnan(H).any():
            print(f"MISMATCH (success/failure flip): {k}")
            sys.exit(1)
        if np.isnan(G).any():
            continue
        e = 0.0 if np.array_equal(G, H) else corner_err(G, H)
        if e > worst[1]:
            worst = (k, e)
    print(f"worst corner deviation vs golden: {worst[1]:.4g} px ({worst[0]})")
    if worst[1] > args.tol:
        print(f"FAIL: exceeds tol {args.tol}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
