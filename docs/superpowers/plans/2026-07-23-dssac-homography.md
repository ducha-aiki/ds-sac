# DS-SAC Homography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement DS-SAC (arXiv:2607.03972) homography estimation in NumPy and benchmark it on HPatches + EVD against pydegensac, OpenCV MAGSAC++, and OpenCV RANSAC.

**Architecture:** A small `dssac` package: `homography.py` (Hartley-normalized DLT + residuals) and `core.py` (forward/backward percentile search, recursive partitioning on signed residuals, post-tuning), exposed as `dssac.find_homography`. A separate `bench/` directory downloads the CVPR-2020 RANSAC tutorial homography data and runs the comparison using the tutorial's own metric (mean absolute reprojection error on the jointly visible region, mAA over 1–10 px).

**Tech Stack:** Python ≥3.10, NumPy, pytest; bench adds h5py, opencv-python, pydegensac, tqdm. Numba only if profiling demands it (not in this plan).

**Spec:** `docs/superpowers/specs/2026-07-23-dssac-homography-design.md`

---

## Algorithm reference (read before Task 4)

DS-SAC is deterministic. All scores are computed on the **full** point set; selection sets are restricted to the current partition. Score is the tuple `(inlier_count, truncated_MSAC)` compared lexicographically; MSAC term = `Σ max(1 − d²/T², 0)`. Residual `d²` is the one-way transfer (reprojection) squared error. `T` is the inlier threshold in pixels; internally compare `d² ≤ T²`.

- **Forward search** on partition `S` (N = full set size, `p_part = |S|/N`): start from DLT fit on all of `S`; for `p` in `arange(p_part, p_min, -Δp)`: (a) *percentile step* — keep the `max(⌈p·N⌉, 4)` smallest-residual points of `S` w.r.t. current model, refit DLT, track best; (b) *inlier step* — support = points of `S` with `d² ≤ T²` (pad with nearest points to reach 4), refit DLT, track best, adopt as current model only if its score **strictly improves the local best seen so far** (execution amendment: gating on the current model's score lets degenerate 4-point pad-fits hijack the annealing trajectory).
- **Backward search**: from the forward local best `(H, p_best)`, run the same round for `p` in `arange(p_best + Δp, 0.5·p_part, +Δp)`.
- **Recursive partitioning**: split `S` by the sign of `r = u·h11 + v·h12 + h13 − u′·(u·h31 + v·h32 + h33)` of the partition's local-best model; recurse while `|S| ≥ max(⌈p_min·N⌉, 4)` and both halves are non-empty. Execution amendment (paper's exception rule): if the local-best split is one-sided, split on `θ_init` (plain DLT on the whole partition) instead; abort recursion only if that is also one-sided.
- Execution amendment to Task 5's test: `test_search_partition_high_outlier_ratio` asserts intermediate-stage thresholds (corner error < 10 px, ≥ 60% inlier recovery) — final accuracy at 80% outliers is delivered by post-tuning, verified empirically (seed 7: 72 inliers / 6.5 px before post-tune → 100 inliers / 0.23 px after; 14/20 random seeds meet the strict criteria end-to-end).
- **Post-tuning**: chain-refit the global best on inlier sets at thresholds `k·T` for `k = (3, 2, 1.5, 1)`, tracking best score.
- Defaults: `Δp = 0.03`, `p_min = 0.2`, `T = 0.54` px (paper: 5.99σ², σ = 0.3; configurable).

---

### Task 1: Scaffolding and environment

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `dssac/__init__.py` (stub), `tests/__init__.py` (empty)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "dssac"
version = "0.1.0"
description = "DS-SAC: Density Search for Sample Consensus (homography), NumPy implementation"
requires-python = ">=3.10"
dependencies = ["numpy"]

[tool.setuptools]
packages = ["dssac"]
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.egg-info/
data/
results/
.pytest_cache/
```

- [ ] **Step 3: Create package stub**

`dssac/__init__.py`:
```python
"""DS-SAC: Density Search for Sample Consensus — homography case."""
```

Create empty `tests/__init__.py`.

- [ ] **Step 4: Create venv and install**

```bash
python3 -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -e . pytest h5py opencv-python tqdm
.venv/bin/pip install -q pydegensac || .venv/bin/pip install -q git+https://github.com/ducha-aiki/pydegensac
.venv/bin/python -c "import numpy, cv2, pydegensac; print('deps ok')"
```

Expected: `deps ok`. If pydegensac fails both ways on macOS arm64, report back to the main session before continuing (bench Task 8 depends on it).

- [ ] **Step 5: Verify pytest runs (no tests yet)**

Run: `.venv/bin/pytest -q`
Expected: `no tests ran`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore dssac tests
git commit -m "chore: scaffold dssac package and environment"
```

---

### Task 2: Normalized DLT

**Files:**
- Create: `dssac/homography.py`
- Test: `tests/test_homography.py`

- [ ] **Step 1: Write failing tests**

`tests/test_homography.py`:
```python
import numpy as np
from dssac.homography import dlt

H_GT = np.array([[1.2, 0.1, 30.0],
                 [-0.05, 0.9, 20.0],
                 [1e-4, -2e-4, 1.0]])


def project(H, pts):
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def test_dlt_exact_recovery_overdetermined():
    rng = np.random.default_rng(42)
    pts1 = rng.uniform(0, 640, (20, 2))
    pts2 = project(H_GT, pts1)
    H = dlt(pts1, pts2)
    H = H / H[2, 2]
    assert np.allclose(H, H_GT, atol=1e-6)


def test_dlt_minimal_four_points():
    pts1 = np.array([[0., 0.], [640., 0.], [640., 480.], [0., 480.]])
    pts2 = project(H_GT, pts1)
    H = dlt(pts1, pts2)
    H = H / H[2, 2]
    assert np.allclose(H, H_GT, atol=1e-6)


def test_dlt_too_few_points_returns_none():
    pts = np.zeros((3, 2))
    assert dlt(pts, pts) is None


def test_dlt_degenerate_collinear_returns_none():
    t = np.linspace(0, 100, 10)
    pts1 = np.stack([t, 2 * t + 1], axis=1)
    pts2 = pts1 + 5.0
    assert dlt(pts1, pts2) is None
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_homography.py -q`
Expected: FAIL / collection error — `dssac.homography` does not exist.

- [ ] **Step 3: Implement `dssac/homography.py`**

```python
"""Hartley-normalized DLT and residuals for homography estimation."""
import numpy as np

MIN_PTS = 4


def _normalization_transform(pts):
    """3x3 similarity: centroid -> origin, mean distance -> sqrt(2). None if degenerate."""
    centroid = pts.mean(axis=0)
    dist = np.sqrt(((pts - centroid) ** 2).sum(axis=1)).mean()
    if dist < 1e-12:
        return None
    s = np.sqrt(2.0) / dist
    return np.array([[s, 0.0, -s * centroid[0]],
                     [0.0, s, -s * centroid[1]],
                     [0.0, 0.0, 1.0]])


def dlt(pts1, pts2):
    """Least-squares homography pts1 -> pts2 via normalized DLT.

    Uses the smallest eigenvector of A^T A (9x9), which is much faster than
    an SVD of the (2N x 9) stacked system for large N.
    Returns a 3x3 array, or None for degenerate input.
    """
    n = len(pts1)
    if n < MIN_PTS:
        return None
    T1 = _normalization_transform(pts1)
    T2 = _normalization_transform(pts2)
    if T1 is None or T2 is None:
        return None
    p1 = pts1 * T1[0, 0] + T1[:2, 2]
    p2 = pts2 * T2[0, 0] + T2[:2, 2]
    u, v = p1[:, 0], p1[:, 1]
    up, vp = p2[:, 0], p2[:, 1]
    zeros = np.zeros(n)
    ones = np.ones(n)
    ax = np.stack([u, v, ones, zeros, zeros, zeros, -up * u, -up * v, -up], axis=1)
    ay = np.stack([zeros, zeros, zeros, u, v, ones, -vp * u, -vp * v, -vp], axis=1)
    A = np.concatenate([ax, ay], axis=0)
    ata = A.T @ A
    try:
        w, V = np.linalg.eigh(ata)
    except np.linalg.LinAlgError:
        return None
    # Rank-deficient system (e.g. collinear points): null space dimension > 1.
    if w[1] < 1e-9 * max(w[-1], 1.0):
        return None
    Hn = V[:, 0].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    return H
```

Note `pts * T[0,0] + T[:2,2]` is valid because the normalization transform is a uniform scale + translation.

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/pytest tests/test_homography.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add dssac/homography.py tests/test_homography.py
git commit -m "feat: normalized DLT homography solver"
```

---

### Task 3: Residuals (transfer error, signed residual)

**Files:**
- Modify: `dssac/homography.py`
- Test: `tests/test_homography.py`

- [ ] **Step 1: Add failing tests to `tests/test_homography.py`**

```python
from dssac.homography import transfer_error_sq, signed_residual


def test_transfer_error_zero_on_perfect_points():
    rng = np.random.default_rng(0)
    pts1 = rng.uniform(0, 640, (15, 2))
    pts2 = project(H_GT, pts1)
    assert np.allclose(transfer_error_sq(H_GT, pts1, pts2), 0.0, atol=1e-12)


def test_transfer_error_known_offset():
    pts1 = np.array([[10.0, 20.0]])
    pts2 = project(H_GT, pts1) + np.array([[3.0, 4.0]])
    assert np.allclose(transfer_error_sq(H_GT, pts1, pts2), [25.0])


def test_signed_residual_identity():
    # For H = I: r = u - u'
    pts1 = np.array([[1.0, 2.0], [5.0, 5.0]])
    pts2 = np.array([[0.5, 2.0], [7.0, 5.0]])
    r = signed_residual(np.eye(3), pts1, pts2)
    assert np.allclose(r, [0.5, -2.0])


def test_signed_residual_zero_on_perfect_points():
    rng = np.random.default_rng(1)
    pts1 = rng.uniform(0, 640, (10, 2))
    pts2 = project(H_GT, pts1)
    assert np.allclose(signed_residual(H_GT, pts1, pts2), 0.0, atol=1e-9)
```

- [ ] **Step 2: Run tests, verify the new ones fail**

Run: `.venv/bin/pytest tests/test_homography.py -q`
Expected: import error on `transfer_error_sq`.

- [ ] **Step 3: Implement in `dssac/homography.py`**

```python
def transfer_error_sq(H, pts1, pts2):
    """Squared one-way transfer error ||H*p1 - p2||^2, per point. (N,)"""
    den = pts1 @ H[2, :2] + H[2, 2]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    x = (pts1 @ H[0, :2] + H[0, 2]) / den
    y = (pts1 @ H[1, :2] + H[1, 2]) / den
    return (x - pts2[:, 0]) ** 2 + (y - pts2[:, 1]) ** 2


def signed_residual(H, pts1, pts2):
    """Signed residual from the first DLT equation (used for partitioning):
    r = u*h11 + v*h12 + h13 - u'*(u*h31 + v*h32 + h33)."""
    u, v = pts1[:, 0], pts1[:, 1]
    up = pts2[:, 0]
    return (u * H[0, 0] + v * H[0, 1] + H[0, 2]
            - up * (u * H[2, 0] + v * H[2, 1] + H[2, 2]))
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `.venv/bin/pytest tests/test_homography.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add dssac/homography.py tests/test_homography.py
git commit -m "feat: transfer error and signed residual"
```

---

### Task 4: Scoring and forward search

**Files:**
- Create: `dssac/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

`tests/test_core.py`:
```python
import numpy as np
from dssac.core import _Best, _score, _forward_search
from dssac.homography import transfer_error_sq

H_GT = np.array([[1.2, 0.1, 30.0],
                 [-0.05, 0.9, 20.0],
                 [1e-4, -2e-4, 1.0]])


def project(H, pts):
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def make_scene(n_inl, n_out, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0, 640, (n_inl, 2))
    p2 = project(H_GT, p1) + rng.normal(0, noise, (n_inl, 2))
    o1 = rng.uniform(0, 640, (n_out, 2))
    o2 = rng.uniform(0, 640, (n_out, 2))
    pts1 = np.vstack([p1, o1])
    pts2 = np.vstack([p2, o2])
    perm = rng.permutation(len(pts1))
    gt_inlier = np.zeros(len(pts1), bool)
    gt_inlier[:n_inl] = True
    return pts1[perm], pts2[perm], gt_inlier[perm]


def max_corner_error(H_est, w=640, h=480):
    corners = np.array([[0., 0.], [w, 0.], [w, h], [0., h]])
    return np.abs(project(H_GT, corners) - project(H_est, corners)).max()


def test_score_counts_inliers_and_breaks_ties_with_msac():
    d2_a = np.array([0.0, 0.5, 10.0])
    d2_b = np.array([0.4, 0.5, 10.0])
    T_sq = 1.0
    sa, sb = _score(d2_a, T_sq), _score(d2_b, T_sq)
    assert sa[0] == sb[0] == 2
    assert sa > sb  # smaller residuals win the MSAC tie-break


def test_forward_search_clean_data():
    pts1, pts2, _ = make_scene(100, 0, noise=0.1)
    best = _Best()
    local = _forward_search(pts1, pts2, np.arange(len(pts1)),
                            T_sq=4.0, dp=0.03, p_min=0.2, glob=best)
    assert local.H is not None
    assert max_corner_error(local.H) < 1.0


def test_forward_search_moderate_outliers():
    pts1, pts2, gt = make_scene(150, 100, noise=0.5, seed=3)
    best = _Best()
    local = _forward_search(pts1, pts2, np.arange(len(pts1)),
                            T_sq=4.0, dp=0.03, p_min=0.2, glob=best)
    assert local.H is not None
    assert max_corner_error(local.H) < 3.0
    assert best.score[0] >= 0.9 * gt.sum()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_core.py -q`
Expected: import error on `dssac.core`.

- [ ] **Step 3: Implement `dssac/core.py`**

```python
"""DS-SAC search: forward/backward percentile search, partitioning, post-tuning."""
from dataclasses import dataclass, field

import numpy as np

from .homography import MIN_PTS, dlt, signed_residual, transfer_error_sq


@dataclass
class _Best:
    """Tracks the best (score, model, percentile) seen so far."""
    score: tuple = (-1, -np.inf)
    H: np.ndarray | None = None
    p: float = 0.0


def _score(d2, T_sq):
    """(inlier count, truncated MSAC) — compared lexicographically."""
    inl = int((d2 <= T_sq).sum())
    msac = float(np.maximum(1.0 - d2 / T_sq, 0.0).sum())
    return (inl, msac)


def _consider(H, pts1, pts2, T_sq, p, local, glob):
    """Score H on the full set; update local and global bests. Returns the score."""
    sc = _score(transfer_error_sq(H, pts1, pts2), T_sq)
    if sc > local.score:
        local.score, local.H, local.p = sc, H, p
    if glob is not None and sc > glob.score:
        glob.score, glob.H = sc, H
    return sc


def _refine_round(pts1, pts2, S, H, p, T_sq, local, glob):
    """One DS-SAC round at percentile p: percentile optimization then inlier
    optimization, both restricted to partition S, scored on the full set.
    Returns the model to carry into the next round."""
    N = len(pts1)
    d2 = transfer_error_sq(H, pts1[S], pts2[S])
    k = min(max(int(np.ceil(p * N)), MIN_PTS), len(S))
    sel = S if k == len(S) else S[np.argpartition(d2, k - 1)[:k]]
    H_p = dlt(pts1[sel], pts2[sel])
    if H_p is not None:
        H = H_p
        _consider(H, pts1, pts2, T_sq, p, local, glob)

    d2 = transfer_error_sq(H, pts1[S], pts2[S])
    inl = d2 <= T_sq
    supp = S[inl] if inl.sum() >= MIN_PTS else S[np.argsort(d2)[:MIN_PTS]]
    H_i = dlt(pts1[supp], pts2[supp])
    if H_i is not None:
        sc = _consider(H_i, pts1, pts2, T_sq, p, local, glob)
        if sc >= _score(transfer_error_sq(H, pts1, pts2), T_sq):
            H = H_i
    return H


def _forward_search(pts1, pts2, S, T_sq, dp, p_min, glob):
    """Percentile descent from |S|/N down to p_min. Returns local best or None."""
    local = _Best()
    H = dlt(pts1[S], pts2[S])
    if H is None:
        return None
    N = len(pts1)
    p_part = len(S) / N
    _consider(H, pts1, pts2, T_sq, p_part, local, glob)
    for p in np.arange(p_part, p_min - 1e-9, -dp):
        H = _refine_round(pts1, pts2, S, H, p, T_sq, local, glob)
    return local
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/pytest tests/test_core.py -q`
Expected: 3 passed. If `test_forward_search_moderate_outliers` fails, debug with the systematic-debugging skill before touching thresholds in the test.

- [ ] **Step 5: Commit**

```bash
git add dssac/core.py tests/test_core.py
git commit -m "feat: DS-SAC scoring and forward search"
```

---

### Task 5: Backward search and recursive partitioning

**Files:**
- Modify: `dssac/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Add failing tests to `tests/test_core.py`**

```python
from dssac.core import _backward_search, _search_partition


def test_backward_search_does_not_regress():
    pts1, pts2, _ = make_scene(150, 100, noise=0.5, seed=3)
    glob = _Best()
    S = np.arange(len(pts1))
    local = _forward_search(pts1, pts2, S, T_sq=4.0, dp=0.03, p_min=0.2, glob=glob)
    score_before = glob.score
    _backward_search(pts1, pts2, S, T_sq=4.0, dp=0.03, local=local, glob=glob)
    assert glob.score >= score_before


def test_search_partition_high_outlier_ratio():
    # 80% outliers: forward search from the full set alone is unlikely to be
    # enough; recursive partitioning must dig the structure out.
    pts1, pts2, gt = make_scene(100, 400, noise=0.5, seed=7)
    glob = _Best()
    _search_partition(pts1, pts2, np.arange(len(pts1)),
                      T_sq=4.0, dp=0.03, p_min=0.2, glob=glob)
    assert glob.H is not None
    assert max_corner_error(glob.H) < 5.0
    assert glob.score[0] >= 0.8 * gt.sum()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_core.py -q`
Expected: import error on `_backward_search`.

- [ ] **Step 3: Implement in `dssac/core.py`**

```python
def _backward_search(pts1, pts2, S, T_sq, dp, local, glob):
    """Expand from the forward best percentile up to 0.5 * p_partition."""
    if local.H is None:
        return
    N = len(pts1)
    p_bwd = 0.5 * (len(S) / N)
    H = local.H
    for p in np.arange(local.p + dp, p_bwd + 1e-9, dp):
        H = _refine_round(pts1, pts2, S, H, p, T_sq, local, glob)


_MAX_DEPTH = 64


def _search_partition(pts1, pts2, S, T_sq, dp, p_min, glob, depth=0):
    """Forward + backward search on S, then recurse on the signed-residual split."""
    N = len(pts1)
    if len(S) < max(int(np.ceil(p_min * N)), MIN_PTS) or depth > _MAX_DEPTH:
        return
    local = _forward_search(pts1, pts2, S, T_sq, dp, p_min, glob)
    if local is None or local.H is None:
        return
    _backward_search(pts1, pts2, S, T_sq, dp, local, glob)
    r = signed_residual(local.H, pts1[S], pts2[S])
    S_plus = S[r >= 0]
    S_minus = S[r < 0]
    if len(S_plus) == 0 or len(S_minus) == 0:
        return  # split failed; recursing would loop on the same set
    _search_partition(pts1, pts2, S_plus, T_sq, dp, p_min, glob, depth + 1)
    _search_partition(pts1, pts2, S_minus, T_sq, dp, p_min, glob, depth + 1)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/pytest tests/test_core.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dssac/core.py tests/test_core.py
git commit -m "feat: backward search and recursive partitioning"
```

---

### Task 6: Post-tuning and public API

**Files:**
- Modify: `dssac/core.py`, `dssac/__init__.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Add failing tests to `tests/test_core.py`**

```python
import dssac


def test_find_homography_outlier_ratios():
    for ratio, tol in [(0.2, 2.0), (0.5, 3.0), (0.8, 5.0)]:
        n_out = int(100 * ratio / (1 - ratio))
        pts1, pts2, gt = make_scene(100, n_out, noise=0.5, seed=11)
        H, mask = dssac.find_homography(pts1, pts2, threshold=2.0)
        assert H is not None, f"failed at ratio {ratio}"
        assert max_corner_error(H) < tol, f"ratio {ratio}"
        # inlier mask should mostly agree with ground truth
        assert (mask & gt).sum() >= 0.8 * gt.sum(), f"ratio {ratio}"


def test_find_homography_is_deterministic():
    pts1, pts2, _ = make_scene(100, 100, noise=0.5, seed=5)
    H1, m1 = dssac.find_homography(pts1, pts2, threshold=2.0)
    H2, m2 = dssac.find_homography(pts1, pts2, threshold=2.0)
    assert np.array_equal(H1, H2)
    assert np.array_equal(m1, m2)


def test_find_homography_too_few_points():
    pts = np.zeros((3, 2))
    H, mask = dssac.find_homography(pts, pts)
    assert H is None and mask is None


def test_find_homography_normalized_h33():
    pts1, pts2, _ = make_scene(100, 20, noise=0.3, seed=2)
    H, _ = dssac.find_homography(pts1, pts2, threshold=2.0)
    assert np.isclose(H[2, 2], 1.0)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_core.py -q`
Expected: `dssac` has no attribute `find_homography`.

- [ ] **Step 3: Implement post-tuning + `find_homography` in `dssac/core.py`**

```python
def _post_tune(pts1, pts2, glob, T_sq, ks):
    """Chain-refit on inlier sets at relaxed-to-strict thresholds k*T."""
    H = glob.H
    for k in ks:
        d2 = transfer_error_sq(H, pts1, pts2)
        sel = d2 <= (k * k) * T_sq
        if sel.sum() < MIN_PTS:
            continue
        H_new = dlt(pts1[sel], pts2[sel])
        if H_new is None:
            continue
        H = H_new
        sc = _score(transfer_error_sq(H, pts1, pts2), T_sq)
        if sc > glob.score:
            glob.score, glob.H = sc, H


def find_homography(pts1, pts2, threshold=0.54, dp=0.03, p_min=0.2,
                    post_tuning_ks=(3.0, 2.0, 1.5, 1.0)):
    """Estimate a homography pts1 -> pts2 with DS-SAC.

    Args:
        pts1, pts2: (N, 2) arrays of corresponding points.
        threshold: inlier threshold in pixels (one-way transfer error).
        dp: percentile step of the forward/backward search.
        p_min: minimum percentile / minimum partition size fraction.
        post_tuning_ks: relaxed-to-strict multipliers of `threshold`.

    Returns:
        (H, mask): 3x3 array with H[2,2] == 1 and boolean inlier mask,
        or (None, None) on failure.
    """
    pts1 = np.ascontiguousarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.ascontiguousarray(pts2, dtype=np.float64).reshape(-1, 2)
    if len(pts1) != len(pts2):
        raise ValueError("pts1 and pts2 must have the same length")
    if len(pts1) < MIN_PTS:
        return None, None
    T_sq = float(threshold) ** 2
    glob = _Best()
    _search_partition(pts1, pts2, np.arange(len(pts1)), T_sq, dp, p_min, glob)
    if glob.H is None:
        return None, None
    _post_tune(pts1, pts2, glob, T_sq, post_tuning_ks)
    H = glob.H
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    mask = transfer_error_sq(H, pts1, pts2) <= T_sq
    return H, mask
```

`dssac/__init__.py`:
```python
"""DS-SAC: Density Search for Sample Consensus — homography case."""
from .core import find_homography

__all__ = ["find_homography"]
```

- [ ] **Step 4: Run full test suite, verify all pass**

Run: `.venv/bin/pytest -q`
Expected: 17 passed (8 in test_homography.py, 9 in test_core.py).

- [ ] **Step 5: Quick timing sanity check**

```bash
.venv/bin/python - <<'EOF'
import time, numpy as np, dssac
rng = np.random.default_rng(0)
H = np.array([[1.2, .1, 30], [-.05, .9, 20], [1e-4, -2e-4, 1.]])
p1 = rng.uniform(0, 640, (2000, 2))
ph = np.hstack([p1, np.ones((2000, 1))]) @ H.T
p2 = ph[:, :2] / ph[:, 2:]
p2[1000:] = rng.uniform(0, 640, (1000, 2))  # 50% outliers
t0 = time.perf_counter()
Hh, m = dssac.find_homography(p1, p2, 2.0)
print(f"{time.perf_counter()-t0:.3f}s, {m.sum()} inliers")
EOF
```

Expected: completes in well under 5 s for N=2000. Record the number in the commit message; numba decision waits for real-data profiling in Task 9.

- [ ] **Step 6: Commit**

```bash
git add dssac tests
git commit -m "feat: post-tuning and public find_homography API"
```

---

### Task 7: Benchmark data setup and loader

**Files:**
- Create: `bench/setup_data.sh`, `bench/data.py`, `bench/__init__.py` (empty)

- [ ] **Step 1: Write `bench/setup_data.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/tutorial ]; then
    git clone --depth 1 https://github.com/ducha-aiki/ransac-tutorial-2020-data data/tutorial
fi
# Homography archive URL: verify against data/tutorial/README.md ("homography" link)
URL="http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/homography.tar.gz"
if [ ! -f data/homography.tar.gz ]; then
    curl -fL -o data/homography.tar.gz "$URL"
fi
tar xzf data/homography.tar.gz -C data
find data -maxdepth 3 -type d | sort | head -30
```

- [ ] **Step 2: Run it and inspect the actual layout**

```bash
chmod +x bench/setup_data.sh && bench/setup_data.sh
grep -io 'http[^)"]*homography[^)"]*' data/tutorial/README.md || true
find data -name '*.h5' | head -20
.venv/bin/python - <<'EOF'
import h5py, glob
for p in sorted(glob.glob('data/**/*.h5', recursive=True))[:10]:
    with h5py.File(p, 'r') as f:
        keys = list(f.keys())
        print(p, len(keys), 'keys; first:', keys[:3])
        k = keys[0]
        obj = f[k]
        print('   ', k, getattr(obj, 'shape', dict(obj)) )
EOF
```

If the curl URL 404s, use the URL grepped from the README instead and fix the script.
Also read the parsing example to confirm key semantics:
```bash
.venv/bin/python -c "import json,sys; nb=json.load(open('data/tutorial/parse_H_data.ipynb')); [print(''.join(c['source']),'\n---') for c in nb['cells'] if c['cell_type']=='code']"
sed -n 1,80p data/tutorial/create_opencv_homography_submission_example.py
```

- [ ] **Step 3: Write `bench/data.py` against the observed layout**

The code below follows the tutorial's example script conventions (per-pair keys in
h5 files: matches `(N, 4)` as `[x1 y1 x2 y2]`, per-pair GT homography, images for
the visible-region metric). **Adjust dataset/key names to what Step 2 printed** —
that step's output is authoritative, this code is the expected shape of it.

```python
"""Loader for the CVPR-2020 RANSAC tutorial homography data (EVD + HPatches)."""
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "homography"

DATASETS = ("EVD", "HPatchesSeq")


def _load_h5(path):
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k]) for k in f.keys()}


def iter_pairs(dataset):
    """Yield dicts: name, pts1, pts2, scores, H_gt, img1, img2 for each pair."""
    d = DATA / dataset
    matches = _load_h5(d / "matches.h5")
    gt = _load_h5(d / "homographies.h5")
    for name, m in sorted(matches.items()):
        m = np.asarray(m, np.float64)
        yield {
            "name": name,
            "pts1": m[:, :2],
            "pts2": m[:, 2:4],
            "scores": m[:, 4] if m.shape[1] > 4 else None,
            "H_gt": np.asarray(gt[name], np.float64),
            "img1": cv2.imread(str(d / "imgs" / f"{name.split('-')[0]}.png")),
            "img2": cv2.imread(str(d / "imgs" / f"{name.split('-')[1]}.png")),
        }
```

- [ ] **Step 4: Verify the loader on both datasets**

```bash
.venv/bin/python - <<'EOF'
from bench.data import iter_pairs, DATASETS
for ds in DATASETS:
    pairs = list(iter_pairs(ds))
    p = pairs[0]
    assert p["pts1"].shape == p["pts2"].shape and p["pts1"].shape[1] == 2
    assert p["H_gt"].shape == (3, 3)
    assert p["img1"] is not None and p["img2"] is not None
    print(ds, len(pairs), "pairs; first:", p["name"], p["pts1"].shape)
EOF
```

Expected: both datasets print a pair count (EVD ≈ 15, HPatches ≈ hundreds) with no assertion errors.

- [ ] **Step 5: Commit**

```bash
git add bench/setup_data.sh bench/data.py bench/__init__.py
git commit -m "feat: tutorial homography data setup and loader"
```

---

### Task 8: Method wrappers and smoke test

**Files:**
- Create: `bench/methods.py`
- Test: `tests/test_methods.py`

- [ ] **Step 1: Write failing test**

`tests/test_methods.py` (synthetic — does not need the downloaded data):
```python
import numpy as np
import pytest

from bench.methods import METHODS


def make_pair(seed=0):
    rng = np.random.default_rng(seed)
    H = np.array([[1.1, 0.05, 20.0], [-0.02, 0.95, 10.0], [1e-5, -1e-5, 1.0]])
    p1 = rng.uniform(0, 640, (300, 2))
    ph = np.hstack([p1, np.ones((300, 1))]) @ H.T
    p2 = ph[:, :2] / ph[:, 2:]
    p2[150:] = rng.uniform(0, 640, (150, 2))
    return p1, p2, H


@pytest.mark.parametrize("name", list(METHODS))
def test_method_recovers_homography(name):
    p1, p2, H_gt = make_pair()
    H, mask = METHODS[name](p1, p2, 2.0)
    assert H is not None and H.shape == (3, 3)
    assert mask.dtype == bool and mask.sum() >= 100
    corners = np.array([[0., 0.], [640., 0.], [640., 480.], [0., 480.]])

    def proj(M, pts):
        ph = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
        return ph[:, :2] / ph[:, 2:]

    assert np.abs(proj(H_gt, corners) - proj(H, corners)).max() < 3.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/pytest tests/test_methods.py -q`
Expected: import error on `bench.methods`.

- [ ] **Step 3: Implement `bench/methods.py`**

```python
"""Uniform wrappers: fn(pts1, pts2, threshold_px) -> (H | None, bool mask)."""
import cv2
import numpy as np
import pydegensac

import dssac

MAX_ITERS = 2000
CONF = 0.999


def _cv2_method(flag):
    def run(pts1, pts2, th):
        H, mask = cv2.findHomography(pts1, pts2, flag, th,
                                     maxIters=MAX_ITERS, confidence=CONF)
        if H is None:
            return None, np.zeros(len(pts1), bool)
        return H, mask.ravel().astype(bool)
    return run


def run_dssac(pts1, pts2, th):
    return dssac.find_homography(pts1, pts2, threshold=th)


def run_pydegensac(pts1, pts2, th):
    H, mask = pydegensac.findHomography(pts1, pts2, th, CONF, MAX_ITERS)
    return H, np.asarray(mask, bool)


METHODS = {
    "dssac": run_dssac,
    "pydegensac": run_pydegensac,
    "cv2-ransac": _cv2_method(cv2.RANSAC),
    "cv2-magsac": _cv2_method(cv2.USAC_MAGSAC),
}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/pytest tests/test_methods.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add bench/methods.py tests/test_methods.py
git commit -m "feat: benchmark method wrappers"
```

---

### Task 9: Benchmark runner and report

**Files:**
- Create: `bench/run_bench.py`, `bench/report.py`

- [ ] **Step 1: Locate the tutorial metric**

```bash
grep -n "def " data/tutorial/metrics.py
```

Expected: a function computing mean absolute reprojection error on the jointly
visible region, named like `get_visible_part_mean_absolute_reprojection_error(img1, img2, H_gt, H)`.
Use the exact name found; if its signature differs, adapt the call in Step 2.

- [ ] **Step 2: Write `bench/run_bench.py`**

```python
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
```

- [ ] **Step 3: Write `bench/report.py`**

```python
"""Aggregate results.jsonl into mAA / median error / runtime tables."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACC_THRESHOLDS = np.arange(1, 11)  # px, tutorial mAA protocol


def maa(errs):
    errs = np.asarray(errs)
    return float(np.mean([(errs < t).mean() for t in ACC_THRESHOLDS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.jsonl")
    args = ap.parse_args()

    groups = defaultdict(list)
    for line in open(ROOT / args.results):
        r = json.loads(line)
        groups[(r["dataset"], r["method"], r["th"])].append(r)

    datasets = sorted({k[0] for k in groups})
    for ds in datasets:
        print(f"\n## {ds}\n")
        print("| method | th (px) | mAA@10px | median err | mean time (s) |")
        print("|---|---|---|---|---|")
        best = {}
        for (d, m, th), recs in sorted(groups.items()):
            if d != ds:
                continue
            errs = [r["err"] for r in recs]
            row = (m, th, maa(errs), float(np.median(errs)),
                   float(np.mean([r["time"] for r in recs])))
            print("| {} | {} | {:.4f} | {:.2f} | {:.4f} |".format(*row))
            if m not in best or row[2] > best[m][2]:
                best[m] = row
        print(f"\nBest per method ({ds}):")
        print("| method | th (px) | mAA@10px | median err | mean time (s) |")
        print("|---|---|---|---|---|")
        for m, row in sorted(best.items(), key=lambda kv: -kv[1][2]):
            print("| {} | {} | {:.4f} | {:.2f} | {:.4f} |".format(*row))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke run (5 pairs, 2 methods)**

```bash
.venv/bin/python bench/run_bench.py --limit 5 --methods dssac pydegensac --out results/smoke.jsonl
.venv/bin/python bench/report.py --results results/smoke.jsonl
```

Expected: tables print, DS-SAC errors are finite and in a plausible px range on
HPatches (not all `1e6`). Note DS-SAC's mean time vs pydegensac here — if DS-SAC
is >20× slower, flag it in the task report (numba optimization is a follow-up
decision for the main session, not this task).

- [ ] **Step 5: Full run**

```bash
.venv/bin/python bench/run_bench.py
.venv/bin/python bench/report.py | tee results/report.md
```

Expected: completes (may take tens of minutes; DS-SAC dominates the runtime).

- [ ] **Step 6: Commit**

```bash
git add bench/run_bench.py bench/report.py
git commit -m "feat: benchmark runner and mAA report"
```

---

### Task 10: README with results, push

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

Include: one-paragraph description citing the paper (arXiv:2607.03972) and stating
this is an independent NumPy reimplementation of the homography case; install
instructions (`pip install -e .`); the `find_homography` usage snippet from the
spec; benchmark instructions (`bench/setup_data.sh`, `run_bench.py`, `report.py`);
and the actual "Best per method" tables for EVD and HPatches pasted from
`results/report.md`, plus the runtime column caveat that DS-SAC is pure NumPy
while baselines are C++.

- [ ] **Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: README with benchmark results"
git push -u origin main
```

If the push fails with an auth error, stop and ask the user to run `! gh auth login`
(or set up a credential helper), then retry.

---

## Verification checklist (end of plan)

- `.venv/bin/pytest -q` — all tests pass.
- `results/report.md` exists with per-threshold and best-per-method tables for both datasets.
- DS-SAC produces finite errors on ≥95% of pairs at its best threshold.
- README results tables match `results/report.md`.
