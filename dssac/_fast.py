"""Numba fast path: the full DS-SAC search pipeline as fused nopython kernels.

Mirrors dssac.core / dssac.homography exactly, with benign deviations:
k-smallest selection uses np.partition + a two-pass scan instead of
argpartition (tie order may differ), floating-point summation order differs,
and the residual denominator clamp treats an exact -0.0 as +0.0. Results can
therefore deviate from the NumPy path in the last bits (verified equivalent
in mAA on the benchmark). Fully deterministic for a fixed input, like the
NumPy path.
"""
import numpy as np
from numba import njit, prange

from .homography import MIN_PTS


@njit(cache=True)
def _eigh9(A):
    """Eigendecomposition of a symmetric 9x9 matrix by cyclic Jacobi rotations.

    Returns (w, V) with eigenvalues ascending and eigenvectors in columns,
    like np.linalg.eigh, but without the scipy/LAPACK dependency. Deterministic;
    converges quadratically (a handful of sweeps for 9x9).
    """
    n = 9
    a = A.copy()
    V = np.eye(n)
    for _sweep in range(50):
        off = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                off += a[i, j] * a[i, j]
        norm = 0.0
        for i in range(n):
            for j in range(n):
                norm += a[i, j] * a[i, j]
        if off <= 1e-30 * (norm if norm > 0.0 else 1.0):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p, q]
                if apq == 0.0:
                    continue
                # Rotation angle zeroing a[p, q] (Golub & Van Loan 8.4).
                tau = (a[q, q] - a[p, p]) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c
                app = a[p, p]
                aqq = a[q, q]
                a[p, p] = app - t * apq
                a[q, q] = aqq + t * apq
                a[p, q] = 0.0
                a[q, p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        akp = a[k, p]
                        akq = a[k, q]
                        a[k, p] = c * akp - s * akq
                        a[p, k] = a[k, p]
                        a[k, q] = s * akp + c * akq
                        a[q, k] = a[k, q]
                for k in range(n):
                    vkp = V[k, p]
                    vkq = V[k, q]
                    V[k, p] = c * vkp - s * vkq
                    V[k, q] = s * vkp + c * vkq
    w = np.empty(n)
    for i in range(n):
        w[i] = a[i, i]
    # Insertion sort ascending, carrying eigenvector columns along.
    for i in range(1, n):
        wi = w[i]
        col = V[:, i].copy()
        j = i - 1
        while j >= 0 and w[j] > wi:
            w[j + 1] = w[j]
            for k in range(n):
                V[k, j + 1] = V[k, j]
            j -= 1
        w[j + 1] = wi
        for k in range(n):
            V[k, j + 1] = col[k]
    return w, V


@njit(cache=True)
def _fit(pts1, pts2, idx):
    """Hartley-normalized least-squares DLT on the subset `idx`.
    Returns (H, ok)."""
    n = idx.shape[0]
    H = np.zeros((3, 3))
    if n < MIN_PTS:
        return H, False
    cx1 = 0.0
    cy1 = 0.0
    cx2 = 0.0
    cy2 = 0.0
    for t in range(n):
        i = idx[t]
        cx1 += pts1[i, 0]
        cy1 += pts1[i, 1]
        cx2 += pts2[i, 0]
        cy2 += pts2[i, 1]
    cx1 /= n
    cy1 /= n
    cx2 /= n
    cy2 /= n
    m1 = 0.0
    m2 = 0.0
    for t in range(n):
        i = idx[t]
        dx = pts1[i, 0] - cx1
        dy = pts1[i, 1] - cy1
        m1 += np.sqrt(dx * dx + dy * dy)
        dx = pts2[i, 0] - cx2
        dy = pts2[i, 1] - cy2
        m2 += np.sqrt(dx * dx + dy * dy)
    m1 /= n
    m2 /= n
    if m1 < 1e-12 or m2 < 1e-12:
        return H, False
    s1 = np.sqrt(2.0) / m1
    s2 = np.sqrt(2.0) / m2

    # A^T A in closed form from point moments (branch-free): with normalized
    # coordinates (u, v) -> (u', v'), the stacked DLT rows are
    # ax = [u, v, 1, 0, 0, 0, -u'u, -u'v, -u'] and
    # ay = [0, 0, 0, u, v, 1, -v'u, -v'v, -v'].
    Su = Sv = Suu = Suv = Svv = 0.0
    Pu = Puu = Puv = Puuu = Puuv = Puvv = 0.0
    Qu = Quu = Quv = Quuu = Quuv = Quvv = 0.0
    R = Ru = Rv = Ruu = Ruv = Rvv = 0.0
    for t in range(n):
        i = idx[t]
        u = s1 * (pts1[i, 0] - cx1)
        v = s1 * (pts1[i, 1] - cy1)
        up = s2 * (pts2[i, 0] - cx2)
        vp = s2 * (pts2[i, 1] - cy2)
        uu = u * u
        uv = u * v
        vv = v * v
        Su += u
        Sv += v
        Suu += uu
        Suv += uv
        Svv += vv
        Pu += up
        Puu += up * u
        Puv += up * v
        Puuu += up * uu
        Puuv += up * uv
        Puvv += up * vv
        Qu += vp
        Quu += vp * u
        Quv += vp * v
        Quuu += vp * uu
        Quuv += vp * uv
        Quvv += vp * vv
        w2 = up * up + vp * vp
        R += w2
        Ru += w2 * u
        Rv += w2 * v
        Ruu += w2 * uu
        Ruv += w2 * uv
        Rvv += w2 * vv
    ata = np.empty((9, 9))
    fn = float(n)
    # upper-left 3x3 blocks (identical for ax and ay parts)
    ata[0, 0] = Suu
    ata[0, 1] = Suv
    ata[0, 2] = Su
    ata[1, 1] = Svv
    ata[1, 2] = Sv
    ata[2, 2] = fn
    for a in range(3):
        for b in range(3):
            ata[a + 3, b + 3] = ata[min(a, b), max(a, b)]
            ata[a, b + 3] = 0.0
    # cross blocks: -(moments weighted by u' and v')
    ata[0, 6] = -Puuu
    ata[0, 7] = -Puuv
    ata[0, 8] = -Puu
    ata[1, 6] = -Puuv
    ata[1, 7] = -Puvv
    ata[1, 8] = -Puv
    ata[2, 6] = -Puu
    ata[2, 7] = -Puv
    ata[2, 8] = -Pu
    ata[3, 6] = -Quuu
    ata[3, 7] = -Quuv
    ata[3, 8] = -Quu
    ata[4, 6] = -Quuv
    ata[4, 7] = -Quvv
    ata[4, 8] = -Quv
    ata[5, 6] = -Quu
    ata[5, 7] = -Quv
    ata[5, 8] = -Qu
    # lower-right block: moments weighted by (u'^2 + v'^2)
    ata[6, 6] = Ruu
    ata[6, 7] = Ruv
    ata[6, 8] = Ru
    ata[7, 7] = Rvv
    ata[7, 8] = Rv
    ata[8, 8] = R
    for a in range(9):
        for b in range(a):
            ata[a, b] = ata[b, a]

    w, V = _eigh9(ata)
    wmax = w[8] if w[8] > 1.0 else 1.0
    if w[1] < 1e-9 * wmax:
        return H, False
    Hn = np.ascontiguousarray(V[:, 0]).reshape(3, 3)
    T1 = np.zeros((3, 3))
    T1[0, 0] = s1
    T1[0, 2] = -s1 * cx1
    T1[1, 1] = s1
    T1[1, 2] = -s1 * cy1
    T1[2, 2] = 1.0
    T2i = np.zeros((3, 3))  # inverse of the target-side similarity
    T2i[0, 0] = 1.0 / s2
    T2i[0, 2] = cx2
    T2i[1, 1] = 1.0 / s2
    T2i[1, 2] = cy2
    T2i[2, 2] = 1.0
    Hh = T2i @ (Hn @ T1)
    for a in range(3):
        for b in range(3):
            if not np.isfinite(Hh[a, b]):
                return H, False
    if abs(Hh[2, 2]) < 1e-12:
        return H, False
    return Hh, True


@njit(cache=True)
def _residuals(H, pts1, pts2):
    """Squared one-way transfer error per point, sign-preserving den clamp."""
    N = pts1.shape[0]
    d2 = np.empty(N)
    for i in range(N):
        den = pts1[i, 0] * H[2, 0] + pts1[i, 1] * H[2, 1] + H[2, 2]
        if den > -1e-12 and den < 1e-12:
            den = 1e-12 if den >= 0.0 else -1e-12
        x = (pts1[i, 0] * H[0, 0] + pts1[i, 1] * H[0, 1] + H[0, 2]) / den
        y = (pts1[i, 0] * H[1, 0] + pts1[i, 1] * H[1, 1] + H[1, 2]) / den
        dx = x - pts2[i, 0]
        dy = y - pts2[i, 1]
        d2[i] = dx * dx + dy * dy
    return d2


@njit(cache=True)
def _msac(d2, T_sq):
    """(inlier count, truncated MSAC term)."""
    inl = 0
    ms = 0.0
    for i in range(d2.shape[0]):
        if d2[i] <= T_sq:
            inl += 1
            ms += 1.0 - d2[i] / T_sq
    return inl, ms


@njit(cache=True)
def _select_k(dS, S, k):
    """Indices (from S) of the k smallest values of dS, deterministically:
    all strictly below the kth order statistic, then ties in index order.
    O(n) via np.partition instead of a full argsort."""
    kth = np.partition(dS.copy(), k - 1)[k - 1]
    sel = np.empty(k, np.int64)
    c = 0
    for t in range(dS.shape[0]):
        if dS[t] < kth:
            sel[c] = S[t]
            c += 1
    for t in range(dS.shape[0]):
        if c >= k:
            break
        if dS[t] == kth:
            sel[c] = S[t]
            c += 1
    return sel


@njit(cache=True)
def _round(pts1, pts2, S, H, d2, p, T_sq,
           l_inl, l_msac, l_H, l_p, g_inl, g_msac, g_H):
    """One percentile + inlier optimization round (mirrors core._refine_round).
    Returns the updated carried state and best trackers."""
    N = pts1.shape[0]
    ns = S.shape[0]
    dS = np.empty(ns)
    for t in range(ns):
        dS[t] = d2[S[t]]
    k = int(np.ceil(p * N))
    if k < MIN_PTS:
        k = MIN_PTS
    if k > ns:
        k = ns
    sel = S if k == ns else _select_k(dS, S, k)
    Hp, ok = _fit(pts1, pts2, sel)
    if ok:
        H = Hp
        d2 = _residuals(H, pts1, pts2)
        inl, ms = _msac(d2, T_sq)
        if inl > l_inl or (inl == l_inl and ms > l_msac):
            l_inl, l_msac, l_H, l_p = inl, ms, H.copy(), p
        if inl > g_inl or (inl == g_inl and ms > g_msac):
            g_inl, g_msac, g_H = inl, ms, H.copy()
        for t in range(ns):
            dS[t] = d2[S[t]]

    cnt = 0
    for t in range(ns):
        if dS[t] <= T_sq:
            cnt += 1
    if cnt >= MIN_PTS:
        supp = np.empty(cnt, np.int64)
        c = 0
        for t in range(ns):
            if dS[t] <= T_sq:
                supp[c] = S[t]
                c += 1
    else:
        supp = _select_k(dS, S, MIN_PTS)
    Hi, ok = _fit(pts1, pts2, supp)
    if ok:
        di = _residuals(Hi, pts1, pts2)
        inl, ms = _msac(di, T_sq)
        prev_inl = l_inl
        prev_ms = l_msac
        if inl > l_inl or (inl == l_inl and ms > l_msac):
            l_inl, l_msac, l_H, l_p = inl, ms, Hi.copy(), p
        if inl > g_inl or (inl == g_inl and ms > g_msac):
            g_inl, g_msac, g_H = inl, ms, Hi.copy()
        if inl > prev_inl or (inl == prev_inl and ms > prev_ms):
            H, d2 = Hi, di
    return H, d2, l_inl, l_msac, l_H, l_p, g_inl, g_msac, g_H


@njit(cache=True)
def search(pts1, pts2, T_sq, dp, p_min, ks):
    """Full DS-SAC pipeline (forward, backward, partitioning, post-tuning).
    Returns (H, found)."""
    N = pts1.shape[0]
    g_inl = -1
    g_msac = -1e300
    g_H = np.zeros((3, 3))
    min_size = int(np.ceil(p_min * N))
    if min_size < MIN_PTS:
        min_size = MIN_PTS

    # Array-based DFS stack. All stacked partitions are pairwise disjoint
    # (children partition their parent), and pops/pushes are LIFO, so a single
    # length-N index buffer used as a segment stack never overflows. The popped
    # segment is copied to S_cur before its space is reused for the children.
    buf = np.empty(N, np.int64)
    for i in range(N):
        buf[i] = i
    _CAP = 256  # >= 2 * max depth (64) + slack
    e_start = np.empty(_CAP, np.int64)
    e_len = np.empty(_CAP, np.int64)
    e_depth = np.empty(_CAP, np.int64)
    e_start[0] = 0
    e_len[0] = N
    e_depth[0] = 0
    n_ent = 1
    S_cur = np.empty(N, np.int64)
    while n_ent > 0:
        n_ent -= 1
        st = e_start[n_ent]
        ln = e_len[n_ent]
        depth = e_depth[n_ent]
        for i in range(ln):
            S_cur[i] = buf[st + i]
        top = st  # the popped entry is always the topmost live segment
        S = S_cur[:ln]
        if ln < min_size or depth > 64:
            continue

        # Forward search (mirrors core._forward_search).
        H, ok = _fit(pts1, pts2, S)
        if not ok:
            continue
        p_part = S.shape[0] / N
        d2 = _residuals(H, pts1, pts2)
        inl, ms = _msac(d2, T_sq)
        l_inl, l_msac, l_H, l_p = inl, ms, H.copy(), p_part
        if inl > g_inl or (inl == g_inl and ms > g_msac):
            g_inl, g_msac, g_H = inl, ms, H.copy()
        t = 0
        p = p_part
        while p > p_min - 1e-9:
            H, d2, l_inl, l_msac, l_H, l_p, g_inl, g_msac, g_H = _round(
                pts1, pts2, S, H, d2, p, T_sq,
                l_inl, l_msac, l_H, l_p, g_inl, g_msac, g_H)
            t += 1
            p = p_part - t * dp

        # Backward search (mirrors core._backward_search).
        p_bwd = 0.5 * p_part
        H = l_H.copy()
        d2 = _residuals(H, pts1, pts2)
        p0 = l_p + dp
        t = 0
        p = p0
        while p < p_bwd + 1e-9:
            H, d2, l_inl, l_msac, l_H, l_p, g_inl, g_msac, g_H = _round(
                pts1, pts2, S, H, d2, p, T_sq,
                l_inl, l_msac, l_H, l_p, g_inl, g_msac, g_H)
            t += 1
            p = p0 + t * dp

        # Signed-residual split on the local best; theta_init fallback.
        split_H = l_H
        for attempt in range(2):
            n_plus = 0
            for t in range(S.shape[0]):
                i = S[t]
                r = (pts1[i, 0] * split_H[0, 0] + pts1[i, 1] * split_H[0, 1]
                     + split_H[0, 2]
                     - pts2[i, 0] * (pts1[i, 0] * split_H[2, 0]
                                     + pts1[i, 1] * split_H[2, 1]
                                     + split_H[2, 2]))
                if r >= 0.0:
                    n_plus += 1
            if 0 < n_plus < S.shape[0]:
                break
            if attempt == 1:
                n_plus = -1
                break
            split_H, ok = _fit(pts1, pts2, S)
            if not ok:
                n_plus = -1
                break
        if n_plus <= 0 or n_plus >= S.shape[0] or n_ent >= _CAP - 2:
            continue
        # Write minus segment first, plus segment on top: the plus partition is
        # then popped first (same order as the recursive NumPy implementation).
        n_minus = ln - n_plus
        cp = top + n_minus
        cm = top
        for t in range(ln):
            i = S[t]
            r = (pts1[i, 0] * split_H[0, 0] + pts1[i, 1] * split_H[0, 1]
                 + split_H[0, 2]
                 - pts2[i, 0] * (pts1[i, 0] * split_H[2, 0]
                                 + pts1[i, 1] * split_H[2, 1]
                                 + split_H[2, 2]))
            if r >= 0.0:
                buf[cp] = i
                cp += 1
            else:
                buf[cm] = i
                cm += 1
        e_start[n_ent] = top
        e_len[n_ent] = n_minus
        e_depth[n_ent] = depth + 1
        e_start[n_ent + 1] = top + n_minus
        e_len[n_ent + 1] = n_plus
        e_depth[n_ent + 1] = depth + 1
        n_ent += 2

    if g_inl < 0:
        return g_H, False

    # Post-tuning (mirrors core._post_tune).
    H = g_H.copy()
    for ki in range(ks.shape[0]):
        lim = ks[ki] * ks[ki] * T_sq
        d2 = _residuals(H, pts1, pts2)
        cnt = 0
        for i in range(N):
            if d2[i] <= lim:
                cnt += 1
        if cnt < MIN_PTS:
            continue
        sel = np.empty(cnt, np.int64)
        c = 0
        for i in range(N):
            if d2[i] <= lim:
                sel[c] = i
                c += 1
        Hn, ok = _fit(pts1, pts2, sel)
        if not ok:
            continue
        H = Hn
        dn = _residuals(H, pts1, pts2)
        inl, ms = _msac(dn, T_sq)
        if inl > g_inl or (inl == g_inl and ms > g_msac):
            g_inl, g_msac, g_H = inl, ms, H.copy()
    return g_H, True


@njit(cache=True, parallel=True)
def search_batch(pts1, pts2, offsets, T_sq, dp, p_min, ks):
    """Run the full pipeline for many pairs in parallel (one prange lane per
    pair). Pairs are packed into flat (M, 2) arrays; pair i spans rows
    offsets[i]:offsets[i+1]. Per-pair results are independent of thread
    scheduling, so the batch is exactly as deterministic as single calls."""
    n_pairs = offsets.shape[0] - 1
    Hs = np.zeros((n_pairs, 3, 3))
    found = np.zeros(n_pairs, np.uint8)
    for ip in prange(n_pairs):
        a = offsets[ip]
        b = offsets[ip + 1]
        if b - a < MIN_PTS:
            continue
        H, ok = search(pts1[a:b], pts2[a:b], T_sq, dp, p_min, ks)
        if ok:
            Hs[ip] = H
            found[ip] = 1
    return Hs, found
