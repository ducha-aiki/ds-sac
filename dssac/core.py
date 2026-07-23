"""DS-SAC search: forward/backward percentile search, partitioning, post-tuning."""
from dataclasses import dataclass, field

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
        sc = _consider(H_i, pts1, pts2, T_sq, p, local, glob)
        if sc >= _score(transfer_error_sq(H, pts1, pts2), T_sq):
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
