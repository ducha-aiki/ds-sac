"""Hartley-normalized DLT and residuals for homography estimation."""
import numpy as np

MIN_PTS = 4


def _normalization_transform(pts):
    """3x3 similarity: centroid -> origin, mean distance -> sqrt(2). None if degenerate."""
    centroid = pts.mean(axis=0)
    dist = np.sqrt(((pts - centroid) ** 2).sum(axis=1)).mean()
    if dist < 1e-12:
        return None
    s = np.sqrt(2.0) / dist
    return np.array([[s, 0.0, -s * centroid[0]],
                     [0.0, s, -s * centroid[1]],
                     [0.0, 0.0, 1.0]])


def dlt(pts1, pts2):
    """Least-squares homography pts1 -> pts2 via normalized DLT.

    Uses the smallest eigenvector of A^T A (9x9), which is much faster than
    an SVD of the (2N x 9) stacked system for large N.
    Returns a 3x3 array, or None for degenerate input.
    """
    n = len(pts1)
    if n < MIN_PTS:
        return None
    T1 = _normalization_transform(pts1)
    T2 = _normalization_transform(pts2)
    if T1 is None or T2 is None:
        return None
    p1 = pts1 * T1[0, 0] + T1[:2, 2]
    p2 = pts2 * T2[0, 0] + T2[:2, 2]
    u, v = p1[:, 0], p1[:, 1]
    up, vp = p2[:, 0], p2[:, 1]
    zeros = np.zeros(n)
    ones = np.ones(n)
    ax = np.stack([u, v, ones, zeros, zeros, zeros, -up * u, -up * v, -up], axis=1)
    ay = np.stack([zeros, zeros, zeros, u, v, ones, -vp * u, -vp * v, -vp], axis=1)
    A = np.concatenate([ax, ay], axis=0)
    ata = A.T @ A
    try:
        w, V = np.linalg.eigh(ata)
    except np.linalg.LinAlgError:
        return None
    # Rank-deficient system (e.g. collinear points): null space dimension > 1.
    if w[1] < 1e-9 * max(w[-1], 1.0):
        return None
    Hn = V[:, 0].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    return H
