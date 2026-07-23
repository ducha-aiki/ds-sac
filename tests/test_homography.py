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
