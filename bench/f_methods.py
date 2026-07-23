"""Uniform F-estimation wrappers: fn(pts1, pts2, threshold_px) -> (F | None, mask)."""
import cv2
import numpy as np
import pydegensac

import dssac

MAX_ITERS = 2000
CONF = 0.999


def _cv2_f_method(flag):
    def run(pts1, pts2, th):
        F, mask = cv2.findFundamentalMat(pts1, pts2, flag, th,
                                         confidence=CONF, maxIters=MAX_ITERS)
        if F is None:
            return None, np.zeros(len(pts1), bool)
        return F[:3], mask.ravel().astype(bool)
    return run


def run_dssac_f(pts1, pts2, th):
    return dssac.find_fundamental(pts1, pts2, threshold=th)


def run_pydegensac_f(pts1, pts2, th):
    F, mask = pydegensac.findFundamentalMatrix(pts1, pts2, th, CONF, MAX_ITERS)
    if F is None:
        return None, np.zeros(len(pts1), bool)
    return F, np.asarray(mask, bool)


F_METHODS = {
    "dssac": run_dssac_f,
    "pydegensac": run_pydegensac_f,
    "cv2-ransac": _cv2_f_method(cv2.FM_RANSAC),
    "cv2-magsac": _cv2_f_method(cv2.USAC_MAGSAC),
}
