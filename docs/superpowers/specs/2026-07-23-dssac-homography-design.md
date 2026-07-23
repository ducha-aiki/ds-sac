# DS-SAC Homography: Design Spec

**Date:** 2026-07-23
**Paper:** DS-SAC: Density Search for Sample Consensus (arXiv:2607.03972), Thapa & Islam
**Repo:** https://github.com/ducha-aiki/ds-sac

## Goal

Reimplement DS-SAC for the homography case in Python (NumPy first, numba only if
profiling on real data shows we are far slower than pydegensac), and benchmark it on
HPatches and EVD from `ransac-tutorial-2020-data` against pydegensac, OpenCV
MAGSAC++, and OpenCV RANSAC.

## Algorithm (full paper pipeline)

DS-SAC is deterministic: no random minimal sampling. It searches dense regions of
residual space.

- **Initialization:** Hartley-normalized DLT fit on all points of the current partition.
- **Forward search:** percentile `p` from 1.0 (fraction of the partition) down to
  `p_min` in steps of `Δp = 0.03`. At each step:
  1. *Percentile optimization:* keep the `⌈p·N⌉` smallest-residual points, refit DLT.
  2. *Inlier optimization:* support set = points with residual ≤ `T`; if smaller than
     4 points, pad with the nearest additional points; refit DLT.
  Track the best model after each refit.
- **Backward search:** from the best forward percentile `p_best`, expand `p` upward in
  steps of `Δp` to `p_bwd = 0.5 · p_partition`, same two optimization steps.
- **Recursive partitioning:** split the partition on the sign of the signed residual of
  its best model, `r = u·h11 + v·h12 + h13 − u′·(u·h31 + v·h32 + h33)` (first DLT
  equation), into `S⁺` (r ≥ 0) and `S⁻` (r < 0); recurse on each while
  `|S|/N ≥ p_min = 0.2`.
- **Post-tuning:** refit the global best model on inlier sets at decreasing thresholds
  `k·T`, `k = [3, 2, 1.5, 1]`. **Assumption:** the paper does not specify the `k`
  sequence; it is configurable.
- **Scoring:** primary = inlier count at `T`; tie-break = truncated MSAC score
  `Σ max(1 − d²/T², 0)`.
- **Residual for scoring/inliers:** one-way transfer (reprojection) error, matching the
  tutorial evaluation convention.
- **Default threshold:** `T = 5.99·σ²` with `σ = 0.3` (≈ 0.54 px), but `T` is a public
  parameter and the benchmark sweeps it.

## Repo layout

```
dssac/
  homography.py   # Hartley-normalized DLT, transfer residuals, signed residual
  core.py         # forward/backward search, recursive partitioning, post-tuning
  __init__.py     # find_homography(pts1, pts2, threshold, ...) -> (H, inlier_mask)
bench/
  setup_data.sh   # clone ransac-tutorial-2020-data, fetch + extract homography.tar.gz
  methods.py      # wrappers: dssac, pydegensac, cv2 RANSAC, cv2 MAGSAC++
  run_bench.py    # HPatches + EVD pairs × methods × threshold grid -> results json
  report.py       # mAA (tutorial metrics.py protocol) + median error + runtime table
tests/
  test_homography.py
  test_core.py
docs/superpowers/specs/
```

## Public API

```python
H, mask = dssac.find_homography(pts1, pts2, threshold=0.54,
                                dp=0.03, p_min=0.2, post_tuning_ks=(3, 2, 1.5, 1))
```

`pts1`, `pts2`: float64 arrays of shape (N, 2). Returns `H` (3×3, `h33 = 1` when
possible) and boolean inlier mask, or `(None, None)` when N < 4 or fits degenerate.

## Benchmark protocol

- Data: `homography.tar.gz` from the tutorial repo — HPatches (~580 pairs) and EVD
  (15 pairs), precomputed correspondences with matching scores, GT homographies.
- Metric: tutorial `metrics.py` mAA over its threshold set, plus median error and mean
  runtime per pair.
- Per-method inlier-threshold sweep: {0.5, 0.75, 1, 2, 4} px; report per-threshold and
  best-per-method.
- pydegensac / OpenCV capped at 2000 iterations (tutorial convention); DS-SAC is
  deterministic, no cap.

## Testing (TDD)

1. `test_homography.py`: DLT recovers a known H exactly from ≥4 clean points; residual
   and signed-residual correctness against manual computation.
2. `test_core.py`: synthetic GT homography + Gaussian noise + outliers at ratios
   {0.2, 0.5, 0.8}; DS-SAC must recover H within tolerance and classify inliers.
3. Benchmark run doubles as integration test.

## Risks

- **pydegensac on macOS arm64:** wheels may be unavailable; fall back to building from
  source, then conda, and flag it if all fail.
- **Underspecified paper details** (post-tuning `k` sequence, exact tie-break usage,
  padding rule): implemented as documented assumptions, configurable where cheap.
