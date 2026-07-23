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
