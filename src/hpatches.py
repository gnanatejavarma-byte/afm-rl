import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hpatches"
SPLIT = ROOT / "configs" / "split.json"

@lru_cache(maxsize=1)
def _index():
    """Map sequence name -> folder (works even if the folder is nested)."""
    return {p.name: p for p in DATA.rglob("*")
            if p.is_dir() and p.name[:2] in ("i_", "v_")}

def load_split():
    return json.loads(SPLIT.read_text())

def list_pairs(seq_names):
    """Every (sequence, k) pair: reference image 1 vs image k, k=2..6."""
    return [(n, k) for n in seq_names for k in range(2, 7)]

def load_pair(seq, k):
    d = _index()[seq]
    img1 = cv2.imread(str(d / "1.ppm"), cv2.IMREAD_GRAYSCALE)
    imgk = cv2.imread(str(d / f"{k}.ppm"), cv2.IMREAD_GRAYSCALE)
    if img1 is None or imgk is None:
        raise FileNotFoundError(f"could not read images for {seq} pair 1-{k}")
    H = np.loadtxt(d / f"H_1_{k}")   # maps points of image 1 into image k
    return img1, imgk, H