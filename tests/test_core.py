import numpy as np
import pytest
import dssac
from dssac.core import _Best, _score, _forward_search, _backward_search, _search_partition
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


def test_backward_search_does_not_regress():
    pts1, pts2, _ = make_scene(150, 100, noise=0.5, seed=3)
    glob = _Best()
    S = np.arange(len(pts1))
    local = _forward_search(pts1, pts2, S, T_sq=4.0, dp=0.03, p_min=0.2, glob=glob)
    score_before = glob.score
    _backward_search(pts1, pts2, S, T_sq=4.0, dp=0.03, local=local, glob=glob)
    assert glob.score >= score_before


def test_backward_search_sweeps_expected_percentile_range(monkeypatch):
    import dssac.core as core
    pts1, pts2, _ = make_scene(50, 0, noise=0.1, seed=0)
    S = np.arange(len(pts1))
    local = _Best()
    local.H = np.eye(3)
    local.p = 0.2
    seen = []

    def fake_refine(pts1_, pts2_, S_, H, d2, p, T_sq, local_, glob_, model_):
        seen.append(p)
        return H, d2

    monkeypatch.setattr(core, "_refine_round", fake_refine)
    _backward_search(pts1, pts2, S, T_sq=4.0, dp=0.1, local=local, glob=_Best())
    # p_part = 1.0, so the sweep should cover local.p + dp .. 0.5 inclusive
    assert np.allclose(seen, [0.3, 0.4, 0.5])


def test_search_partition_high_outlier_ratio():
    # 80% outliers: forward search from the full set alone is unlikely to be
    # enough; recursive partitioning must dig the structure out. Final accuracy
    # at this contamination level is delivered by post-tuning (find_homography);
    # this stage only needs to land in the true model's basin of attraction.
    pts1, pts2, gt = make_scene(100, 400, noise=0.5, seed=7)
    glob = _Best()
    _search_partition(pts1, pts2, np.arange(len(pts1)),
                      T_sq=4.0, dp=0.03, p_min=0.2, glob=glob)
    assert glob.H is not None
    assert max_corner_error(glob.H) < 10.0
    assert glob.score[0] >= 0.6 * gt.sum()


import dssac.core

BACKENDS = ["numpy"] + (["numba"] if dssac.core._fast is not None else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_homography_outlier_ratios(backend):
    for ratio, tol in [(0.2, 2.0), (0.5, 3.0), (0.8, 5.0)]:
        n_out = int(100 * ratio / (1 - ratio))
        pts1, pts2, gt = make_scene(100, n_out, noise=0.5, seed=11)
        H, mask = dssac.find_homography(pts1, pts2, threshold=2.0, backend=backend)
        assert H is not None, f"failed at ratio {ratio}"
        assert max_corner_error(H) < tol, f"ratio {ratio}"
        # inlier mask should mostly agree with ground truth
        assert (mask & gt).sum() >= 0.8 * gt.sum(), f"ratio {ratio}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_homography_is_deterministic(backend):
    pts1, pts2, _ = make_scene(100, 100, noise=0.5, seed=5)
    H1, m1 = dssac.find_homography(pts1, pts2, threshold=2.0, backend=backend)
    H2, m2 = dssac.find_homography(pts1, pts2, threshold=2.0, backend=backend)
    assert np.array_equal(H1, H2)
    assert np.array_equal(m1, m2)


def test_find_homography_too_few_points():
    pts = np.zeros((3, 2))
    H, mask = dssac.find_homography(pts, pts)
    assert H is None and mask is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_homography_normalized_h33(backend):
    pts1, pts2, _ = make_scene(100, 20, noise=0.3, seed=2)
    H, _ = dssac.find_homography(pts1, pts2, threshold=2.0, backend=backend)
    assert np.isclose(H[2, 2], 1.0)


def test_find_homography_rejects_wrong_shape():
    good = np.zeros((10, 2))
    bad = np.zeros((10, 3))
    with pytest.raises(ValueError):
        dssac.find_homography(bad, bad)
    with pytest.raises(ValueError):
        dssac.find_homography(good, np.zeros((11, 2)))


@pytest.mark.skipif(dssac.core._fast is None, reason="numba not installed")
def test_jacobi_eigh_matches_numpy():
    from dssac._fast import _jacobi_eigh
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.normal(size=(20, 9))
        ata = A.T @ A
        w, V = _jacobi_eigh(ata)
        w_np = np.linalg.eigvalsh(ata)
        assert np.all(np.diff(w) >= 0)
        assert np.allclose(w, w_np, rtol=1e-9, atol=1e-9 * abs(w_np[-1]))
        # eigenpair property for the vector we actually use (smallest)
        assert np.allclose(ata @ V[:, 0], w[0] * V[:, 0],
                           atol=1e-8 * abs(w_np[-1]))


@pytest.mark.skipif(dssac.core._fast is None, reason="numba not installed")
def test_find_homographies_batch_matches_single():
    scenes = [make_scene(100, n_out, noise=0.5, seed=s)
              for s, n_out in [(0, 50), (1, 100), (2, 200), (3, 20), (4, 0)]]
    batch = dssac.find_homographies([s[0] for s in scenes],
                                    [s[1] for s in scenes], threshold=2.0)
    for (pts1, pts2, _), (Hb, mb) in zip(scenes, batch):
        Hs, ms = dssac.find_homography(pts1, pts2, threshold=2.0,
                                       backend="numba")
        assert np.array_equal(Hb, Hs)
        assert np.array_equal(mb, ms)


@pytest.mark.skipif(dssac.core._fast is None, reason="numba not installed")
def test_find_homographies_handles_failures_and_empty():
    assert dssac.find_homographies([], []) == []
    pts1, pts2, _ = make_scene(100, 50, noise=0.5, seed=0)
    tiny = np.zeros((3, 2))
    out = dssac.find_homographies([tiny, pts1], [tiny, pts2], threshold=2.0)
    assert out[0] == (None, None)
    assert out[1][0] is not None
