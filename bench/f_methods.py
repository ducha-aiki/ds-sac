"""Uniform F-estimation wrappers: fn(pts1, pts2, threshold_px, scores=None)
-> (F | None, mask). `scores` are SNN ratios (lower = better); only PROSAC-based
methods use them, everyone else ignores the argument."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pydegensac

import dssac

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "rb2025"))
try:
    from ransac_benchmark_imc.vibesac import \
        ransac_fundamental_loransac_numba_refactored as _vibesac
except ImportError:
    _vibesac = None

MAX_ITERS = 2000
CONF = 0.999


def _cv2_f_method(flag):
    def run(pts1, pts2, th, scores=None):
        F, mask = cv2.findFundamentalMat(pts1, pts2, flag, th,
                                         confidence=CONF, maxIters=MAX_ITERS)
        if F is None:
            return None, np.zeros(len(pts1), bool)
        return F[:3], mask.ravel().astype(bool)
    return run


def run_dssac_f(pts1, pts2, th, scores=None):
    return dssac.find_fundamental(pts1, pts2, threshold=th)


def run_dssac_f_pmin01(pts1, pts2, th, scores=None):
    # Lower partition/percentile floor: tolerates inlier ratios down to ~10%
    # at the cost of a deeper search tree.
    return dssac.find_fundamental(pts1, pts2, threshold=th, p_min=0.1)


def run_pydegensac_f(pts1, pts2, th, scores=None):
    F, mask = pydegensac.findFundamentalMatrix(pts1, pts2, th, CONF, MAX_ITERS)
    if F is None:
        return None, np.zeros(len(pts1), bool)
    return F, np.asarray(mask, bool)


def _vibesac_method(max_trials=MAX_ITERS, use_prosac=True):
    def run(pts1, pts2, th, scores=None):
        # vibesac's minimal-sample rejection loop spins forever if asked for
        # 7 unique indices out of fewer than 7 points; guard before it hangs.
        if len(pts1) < 7:
            return None, np.zeros(len(pts1), bool)
        # PROSAC assumes quality-ordered input: sort by SNN ratio ascending
        # (best first) and map the returned mask back to the input order.
        # use_prosac=False forces plain uniform sampling even if scores are
        # supplied, for an ablation against the PROSAC-ordered variant.
        prosac = use_prosac and scores is not None
        order = np.argsort(scores, kind="stable") if prosac \
            else np.arange(len(pts1))
        F, inl, n_inl, _score, _trials = _vibesac(
            np.ascontiguousarray(pts1[order]),
            np.ascontiguousarray(pts2[order]),
            th, min_samples=7, max_trials=max_trials, p_success=CONF,
            use_prosac=prosac)
        if F is None or n_inl < 8 or not np.any(F):
            return None, np.zeros(len(pts1), bool)
        mask = np.zeros(len(pts1), bool)
        mask[order] = inl.astype(bool)
        return F, mask
    return run


F_METHODS = {
    "dssac": run_dssac_f,
    "dssac-pmin0.1": run_dssac_f_pmin01,
    "pydegensac": run_pydegensac_f,
    "cv2-ransac": _cv2_f_method(cv2.FM_RANSAC),
    "cv2-magsac": _cv2_f_method(cv2.USAC_MAGSAC),
}
if _vibesac is not None:
    F_METHODS["vibesac"] = _vibesac_method()
    F_METHODS["vibesac-noprosac"] = _vibesac_method(use_prosac=False)


def _dssac_f_budget(p_min):
    def factory(dp):
        def run(pts1, pts2, th, scores=None):
            return dssac.find_fundamental(pts1, pts2, threshold=th, dp=dp,
                                          p_min=p_min)
        return run
    return factory


def _pydegensac_f_budget(max_iters):
    def run(pts1, pts2, th, scores=None):
        F, mask = pydegensac.findFundamentalMatrix(pts1, pts2, th, CONF,
                                                   max_iters)
        if F is None:
            return None, np.zeros(len(pts1), bool)
        return F, np.asarray(mask, bool)
    return run


def _vibesac_f_budget(use_prosac=True):
    def factory(max_trials):
        return _vibesac_method(max_trials, use_prosac=use_prosac)
    return factory


def _cv2_f_budget(flag):
    def factory(max_iters):
        def run(pts1, pts2, th, scores=None):
            F, mask = cv2.findFundamentalMat(pts1, pts2, flag, th,
                                             confidence=CONF,
                                             maxIters=max_iters)
            if F is None:
                return None, np.zeros(len(pts1), bool)
            return F[:3], mask.ravel().astype(bool)
        return run
    return factory


# Compute-budget knob per method for the time-mAA curve (dp for DS-SAC,
# iteration cap for the RANSAC family), plus each method's tuned SNN ratio
# threshold from the results_f_snn.jsonl grid.
_ITER_BUDGETS_F = (10, 25, 100, 400, 1600, 6400)
_DP_BUDGETS = (0.3, 0.2, 0.12, 0.06, 0.03, 0.015)
F_BUDGETS = {
    "dssac": (_DP_BUDGETS, _dssac_f_budget(0.2), 0.75),
    "dssac-pmin0.1": (_DP_BUDGETS, _dssac_f_budget(0.1), 0.75),
    "pydegensac": (_ITER_BUDGETS_F, _pydegensac_f_budget, 0.8),
    "cv2-ransac": (_ITER_BUDGETS_F, _cv2_f_budget(cv2.FM_RANSAC), 0.8),
    "cv2-magsac": (_ITER_BUDGETS_F, _cv2_f_budget(cv2.USAC_MAGSAC), 0.75),
}
if _vibesac is not None:
    F_BUDGETS["vibesac"] = (_ITER_BUDGETS_F, _vibesac_f_budget(), 0.85)
    F_BUDGETS["vibesac-noprosac"] = (_ITER_BUDGETS_F,
                                     _vibesac_f_budget(use_prosac=False), 0.8)
