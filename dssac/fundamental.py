"""Normalized 8-point fundamental-matrix solver and Sampson residuals."""
import numpy as np

from .homography import _normalization_transform

MIN_PTS_F = 8


def fit_fundamental(pts1, pts2):
    """Least-squares fundamental matrix pts1 -> pts2 (x2^T F x1 = 0) via the
    Hartley-normalized 8-point algorithm with rank-2 enforcement.
    Returns a 3x3 array (unit Frobenius norm), or None for degenerate input."""
    n = len(pts1)
    if n < MIN_PTS_F:
        return None
    T1 = _normalization_transform(pts1)
    T2 = _normalization_transform(pts2)
    if T1 is None or T2 is None:
        return None
    p1 = pts1 * T1[0, 0] + T1[:2, 2]
    p2 = pts2 * T2[0, 0] + T2[:2, 2]
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    A = np.empty((n, 9))
    A[:, 0] = x2 * x1
    A[:, 1] = x2 * y1
    A[:, 2] = x2
    A[:, 3] = y2 * x1
    A[:, 4] = y2 * y1
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1
    A[:, 8] = 1.0
    ata = A.T @ A
    try:
        w, V = np.linalg.eigh(ata)
    except np.linalg.LinAlgError:
        return None
    if w[1] < 1e-9 * max(w[-1], 1.0):
        return None
    Fn = V[:, 0].reshape(3, 3)
    # Rank-2 enforcement without an SVD: project out the right null direction
    # v0 (smallest eigenvector of Fn^T Fn), i.e. subtract the smallest
    # rank-1 component: Fn (I - v0 v0^T) = U diag(s1, s2, 0) V^T.
    wf, Vf = np.linalg.eigh(Fn.T @ Fn)
    v0 = Vf[:, 0]
    Fn = Fn - np.outer(Fn @ v0, v0)
    F = T2.T @ Fn @ T1
    norm = np.linalg.norm(F)
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return F / norm


def _epipolar_terms(F, pts1, pts2):
    """(x2^T F x1, F x1 components, F^T x2 components), vectorized."""
    Fx1 = pts1 @ F[:, :2].T + F[:, 2]          # (N, 3): rows F @ [x1, y1, 1]
    Ftx2 = pts2 @ F[:2, :] + F[2, :]           # (N, 3): rows F^T @ [x2, y2, 1]
    num = (Fx1[:, 0] * pts2[:, 0] + Fx1[:, 1] * pts2[:, 1] + Fx1[:, 2])
    return num, Fx1, Ftx2


def sampson_sq(F, pts1, pts2):
    """Squared Sampson distance per correspondence. (N,)"""
    num, Fx1, Ftx2 = _epipolar_terms(F, pts1, pts2)
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    den = np.where(den < 1e-15, 1e-15, den)
    return num * num / den


def signed_epipolar(F, pts1, pts2):
    """Signed epipolar constraint x2^T F x1 (used for partitioning)."""
    num, _, _ = _epipolar_terms(F, pts1, pts2)
    return num
