"""Loader for the CVPR-2020 RANSAC tutorial homography data (EVD + HPatches).

Data layout on disk (after running ``bench/setup_data.sh``)::

    data/homography/EVD/{val,test}/{matches.h5,match_conf.h5,Hgt.h5,imgs/{1,2}/<name>.png}
    data/homography/HPatchesSeq/{val,test}/{matches.h5,match_conf.h5,Hgt.h5,imgs/<seq>/{1..6}.ppm}

Only the ``val`` split ships ground-truth homographies (``Hgt.h5``); the
``test`` split has correspondences and images but no GT, so it is unusable
for benchmarking and is skipped entirely.

Correspondence convention (verified against the tutorial's
``parse_H_data.ipynb`` and ``create_opencv_homography_submission_example.py``):
each row of ``matches.h5[key]`` is ``[x1, y1, x2, y2]`` with columns 0:2 in
image 1 and 2:4 in image 2. ``match_conf.h5[key]`` holds a per-correspondence
score (lower is better, e.g. SNN ratio -- matches the tutorial's convention of
keeping matches with ``score <= threshold``). ``Hgt.h5[key]`` is the 3x3
ground-truth homography that maps image-1 pixel coordinates to image-2 pixel
coordinates (``metrics.py`` warps image 1's mask into image 2 with ``H_gt``
directly, i.e. ``H_gt`` is *not* inverted).
"""
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "homography"

DATASETS = ("EVD", "HPatchesSeq")

# The only split with ground-truth homographies available (see module docstring).
_SPLIT = "val"


def _load_h5_dict(path):
    """Load an hdf5 file of the tutorial's {key: array} format into a dict."""
    d = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            d[key] = f[key][()]
    return d


def _image_paths(dataset, key):
    """Resolve the (img1_path, img2_path) for a given pair key, following the
    naming convention used by the tutorial's ``utils.get_h_imgpair``."""
    imgs_dir = DATA / dataset / _SPLIT / "imgs"
    if dataset == "EVD":
        stem = key.split("-")[0]
        return imgs_dir / "1" / f"{stem}.png", imgs_dir / "2" / f"{stem}.png"
    if dataset == "HPatchesSeq":
        seq = key[:-4]
        idx2 = key[-1]
        return imgs_dir / seq / "1.ppm", imgs_dir / seq / f"{idx2}.ppm"
    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")


def iter_pairs(dataset):
    """Yield dicts with keys: name, pts1 (N,2 float64), pts2 (N,2 float64),
    scores (N,) or None, H_gt (3,3 float64), img1, img2 (BGR arrays)."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")

    split_dir = DATA / dataset / _SPLIT
    matches = _load_h5_dict(split_dir / "matches.h5")
    conf = _load_h5_dict(split_dir / "match_conf.h5")
    hgt = _load_h5_dict(split_dir / "Hgt.h5")

    for key in sorted(hgt.keys()):
        m = np.asarray(matches[key], dtype=np.float64)
        pts1 = np.ascontiguousarray(m[:, 0:2])
        pts2 = np.ascontiguousarray(m[:, 2:4])
        scores = np.asarray(conf[key], dtype=np.float64).reshape(-1)
        H_gt = np.asarray(hgt[key], dtype=np.float64)

        img1_path, img2_path = _image_paths(dataset, key)
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        if img1 is None or img2 is None:
            raise FileNotFoundError(
                f"Could not read images for {dataset}/{key}: {img1_path}, {img2_path}"
            )

        yield {
            "name": key,
            "pts1": pts1,
            "pts2": pts2,
            "scores": scores,
            "H_gt": H_gt,
            "img1": img1,
            "img2": img2,
        }
