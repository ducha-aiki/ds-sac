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
