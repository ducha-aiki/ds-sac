"""DS-SAC search: forward/backward percentile search, partitioning, post-tuning."""
from dataclasses import dataclass

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
        prev_best = local.score
        sc = _consider(H_i, pts1, pts2, T_sq, p, local, glob)
        if sc > prev_best:
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
        # One-sided split on the local best: fall back to splitting on the
        # plain LSQ fit of the whole partition (theta_init), which sits
        # "between" structures rather than overfitting to a handful of points.
        H_init = dlt(pts1[S], pts2[S])
        if H_init is None:
            return
        r = signed_residual(H_init, pts1[S], pts2[S])
        S_plus = S[r >= 0]
        S_minus = S[r < 0]
        if len(S_plus) == 0 or len(S_minus) == 0:
            return  # split failed; recursing would loop on the same set
    _search_partition(pts1, pts2, S_plus, T_sq, dp, p_min, glob, depth + 1)
    _search_partition(pts1, pts2, S_minus, T_sq, dp, p_min, glob, depth + 1)


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
    pts1 = np.ascontiguousarray(pts1, dtype=np.float64)
    pts2 = np.ascontiguousarray(pts2, dtype=np.float64)
    if pts1.ndim != 2 or pts1.shape[1] != 2 or pts1.shape != pts2.shape:
        raise ValueError("pts1 and pts2 must both have shape (N, 2)")
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
