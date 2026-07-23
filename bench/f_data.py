"""Loader for one PhotoTourism validation scene (RANSAC tutorial 2020 F data)."""
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F_DATA = ROOT / "data" / "f_data"

SCENES = ("st_peters_square",)


def _load_h5(path):
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k]) for k in f.keys()}


def iter_pairs_f(scene):
    """Yield dicts per pair: name, pts1, pts2 (N,2 float64), scores (SNN ratio,
    lower is better), K1, K2, R1, R2, T1, T2 (ground-truth calibration/pose)."""
    d = F_DATA / scene
    matches = _load_h5(d / "matches.h5")
    conf = _load_h5(d / "match_conf.h5")
    K1_K2 = _load_h5(d / "K1_K2.h5")
    R = _load_h5(d / "R.h5")
    T = _load_h5(d / "T.h5")
    for name in sorted(matches):
        m = np.asarray(matches[name], np.float64)
        id1, id2 = name.split("-")
        yield {
            "name": name,
            "pts1": m[:, :2],
            "pts2": m[:, 2:4],
            "scores": np.asarray(conf[name], np.float64),
            "K1": np.asarray(K1_K2[name][0][0], np.float64),
            "K2": np.asarray(K1_K2[name][0][1], np.float64),
            "R1": np.asarray(R[id1], np.float64),
            "R2": np.asarray(R[id2], np.float64),
            "T1": np.asarray(T[id1], np.float64).reshape(3),
            "T2": np.asarray(T[id2], np.float64).reshape(3),
        }
