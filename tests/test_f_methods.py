import numpy as np
import pytest

from bench.f_methods import F_METHODS
from tests.test_fundamental import make_two_view
from dssac.fundamental import sampson_sq


@pytest.mark.parametrize("name", list(F_METHODS))
def test_f_method_recovers_epipolar_geometry(name):
    pts1, pts2, F_gt, gt = make_two_view(300, noise=0.5, seed=1, n_out=150)
    F, mask = F_METHODS[name](pts1, pts2, 1.0)
    assert F is not None and F.shape == (3, 3)
    assert mask.dtype == bool and mask.sum() >= 150
    med = np.median(np.sqrt(sampson_sq(F, pts1[gt], pts2[gt])))
    assert med < 1.0, f"median GT-inlier Sampson {med:.3f}px"
