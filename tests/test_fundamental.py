import numpy as np
import pytest

import dssac
import dssac.core
from dssac.fundamental import fit_fundamental, sampson_sq, signed_epipolar


def make_two_view(n, noise=0.0, seed=0, n_out=0):
    """Synthetic calibrated two-view scene; returns pts1, pts2, F_gt, gt_inlier."""
    rng = np.random.default_rng(seed)
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    R = np.array([[0.9962, 0.0, 0.0872], [0.0, 1.0, 0.0], [-0.0872, 0.0, 0.9962]])
    t = np.array([0.5, 0.05, 0.1])
    X = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n),
                         rng.uniform(4, 10, n)])
    x1 = X @ K.T
    x1 = x1[:, :2] / x1[:, 2:]
    x2 = (X @ R.T + t) @ K.T
    x2 = x2[:, :2] / x2[:, 2:]
    x1 += rng.normal(0, noise, x1.shape)
    x2 += rng.normal(0, noise, x2.shape)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ tx @ R @ Kinv
    F = F / np.linalg.norm(F)
    o1 = rng.uniform(0, 640, (n_out, 2))
    o2 = rng.uniform(0, 480, (n_out, 2))
    pts1 = np.vstack([x1, o1])
    pts2 = np.vstack([x2, o2])
    gt = np.zeros(len(pts1), bool)
    gt[:n] = True
    perm = rng.permutation(len(pts1))
    return pts1[perm], pts2[perm], F, gt[perm]


def f_dist(Fa, Fb):
    """Scale/sign-invariant distance between fundamental matrices."""
    a = Fa / np.linalg.norm(Fa)
    b = Fb / np.linalg.norm(Fb)
    return min(np.abs(a - b).max(), np.abs(a + b).max())


def test_fit_fundamental_exact_recovery():
    pts1, pts2, F_gt, _ = make_two_view(50)
    F = fit_fundamental(pts1, pts2)
    assert F is not None
    assert f_dist(F, F_gt) < 1e-6
    assert np.linalg.matrix_rank(F, tol=1e-9 * np.linalg.norm(F)) == 2


def test_fit_fundamental_minimal_eight_points():
    pts1, pts2, F_gt, _ = make_two_view(8, seed=3)
    F = fit_fundamental(pts1, pts2)
    assert F is not None
    assert f_dist(F, F_gt) < 1e-6


def test_fit_fundamental_too_few_returns_none():
    pts1, pts2, _, _ = make_two_view(7)
    assert fit_fundamental(pts1, pts2) is None


def test_sampson_zero_on_perfect_points():
    pts1, pts2, F_gt, _ = make_two_view(30, seed=1)
    assert np.all(sampson_sq(F_gt, pts1, pts2) < 1e-12)


def test_sampson_positive_off_epipolar():
    pts1, pts2, F_gt, _ = make_two_view(30, seed=2)
    d2 = sampson_sq(F_gt, pts1 + np.array([3.0, 0.0]), pts2)
    assert np.all(d2 > 1e-3)


def test_signed_epipolar_signs():
    pts1, pts2, F_gt, _ = make_two_view(30, seed=4)
    r = signed_epipolar(F_gt, pts1, pts2)
    assert np.allclose(r, 0.0, atol=1e-9)
    r_shift = signed_epipolar(F_gt, pts1, pts2 + np.array([0.0, 5.0]))
    assert (r_shift > 0).any() and (r_shift < 0).any() or np.abs(r_shift).max() > 1e-6


F_BACKENDS = ["numpy"] + (["numba"] if dssac.core._fast is not None else [])


@pytest.mark.parametrize("backend", F_BACKENDS)
def test_find_fundamental_outlier_ratios(backend):
    # Entrywise distance to F_gt is a poor criterion under noise (the epipole
    # is weakly constrained), so assert epipolar quality instead: the median
    # Sampson error of the true inliers under the estimate must be close to
    # what an oracle least-squares fit on the true inliers achieves (~0.34 px
    # at this noise level).
    for ratio in (0.2, 0.5):
        n_out = int(200 * ratio / (1 - ratio))
        pts1, pts2, F_gt, gt = make_two_view(200, noise=0.5, seed=7, n_out=n_out)
        F, mask = dssac.find_fundamental(pts1, pts2, threshold=1.0, backend=backend)
        assert F is not None, f"failed at ratio {ratio}"
        med = np.median(np.sqrt(sampson_sq(F, pts1[gt], pts2[gt])))
        assert med < 0.6, f"ratio {ratio}: median GT-inlier Sampson {med:.3f}px"
        assert (mask & gt).sum() >= 0.8 * gt.sum(), f"ratio {ratio}"


@pytest.mark.parametrize("backend", F_BACKENDS)
def test_find_fundamental_deterministic(backend):
    pts1, pts2, _, _ = make_two_view(150, noise=0.5, seed=5, n_out=100)
    F1, m1 = dssac.find_fundamental(pts1, pts2, threshold=1.0, backend=backend)
    F2, m2 = dssac.find_fundamental(pts1, pts2, threshold=1.0, backend=backend)
    assert np.array_equal(F1, F2) and np.array_equal(m1, m2)


@pytest.mark.skipif(dssac.core._fast is None, reason="numba not installed")
def test_find_fundamentals_batch_matches_single():
    scenes = [make_two_view(150, noise=0.5, seed=s, n_out=80) for s in range(3)]
    batch = dssac.find_fundamentals([s[0] for s in scenes], [s[1] for s in scenes],
                                    threshold=1.0)
    for (p1, p2, _, _), (Fb, mb) in zip(scenes, batch):
        Fs, ms = dssac.find_fundamental(p1, p2, threshold=1.0, backend="numba")
        assert np.array_equal(Fb, Fs) and np.array_equal(mb, ms)
