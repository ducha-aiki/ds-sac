"""Uniform wrappers: fn(pts1, pts2, threshold_px) -> (H | None, bool mask)."""
import cv2
import numpy as np
import pydegensac

import dssac

MAX_ITERS = 2000
CONF = 0.999


def _cv2_method(flag, max_iters=MAX_ITERS):
    def run(pts1, pts2, th):
        H, mask = cv2.findHomography(pts1, pts2, flag, th,
                                     maxIters=max_iters, confidence=CONF)
        if H is None:
            return None, np.zeros(len(pts1), bool)
        return H, mask.ravel().astype(bool)
    return run


def _dssac_method(dp=0.03):
    def run(pts1, pts2, th):
        return dssac.find_homography(pts1, pts2, threshold=th, dp=dp)
    return run


def _pydegensac_method(max_iters=MAX_ITERS):
    def run(pts1, pts2, th):
        H, mask = pydegensac.findHomography(pts1, pts2, th, CONF, max_iters)
        return H, np.asarray(mask, bool)
    return run


METHODS = {
    "dssac": _dssac_method(),
    "pydegensac": _pydegensac_method(),
    "cv2-ransac": _cv2_method(cv2.RANSAC),
    "cv2-magsac": _cv2_method(cv2.USAC_MAGSAC),
}

# Compute-budget knob per method for the time-mAA curve: iteration cap for the
# RANSAC family, percentile step dp for DS-SAC (deterministic — finer dp means
# more refit rounds).
_ITER_BUDGETS = (10, 25, 100, 400, 1600, 6400)
BUDGETS = {
    "dssac": ((0.3, 0.2, 0.12, 0.06, 0.03, 0.015), _dssac_method),
    "pydegensac": (_ITER_BUDGETS, _pydegensac_method),
    "cv2-ransac": (_ITER_BUDGETS, lambda it: _cv2_method(cv2.RANSAC, it)),
    "cv2-magsac": (_ITER_BUDGETS, lambda it: _cv2_method(cv2.USAC_MAGSAC, it)),
}
