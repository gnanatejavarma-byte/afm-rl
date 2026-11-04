import time
from collections import OrderedDict
from functools import lru_cache
import cv2
import numpy as np
from cv_compat import make
from pathlib import Path

# ---- keypoint budget ----------------------------------------------------
N_MIN, N_START, N_MAX = 100, 500, 4000
MAX_POOL = 5000          # most candidates we ask any detector for
POOL_CACHE_SIZE = 128    # (image, preset) pools kept in memory

# ---- learned preset support (XFeat, ALIKED via kornia) ------------------
LEARNED_PRESETS = {"XFeat", "ALIKED"}

try:
    import torch as _torch
    import kornia.feature as _KF
    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False

# Lazily-loaded kornia model objects — populated on first call
_LEARNED_MODELS: dict = {}


def _get_learned_model(preset: str):
    """Return the cached kornia model for a learned preset, loading it once."""
    if not _KORNIA_AVAILABLE:
        raise ImportError(
            f"Preset '{preset}' requires kornia and torch. "
            "Install with: pip install kornia kornia-rs"
        )
    if preset not in _LEARNED_MODELS:
        if preset == "XFeat":
            # KF.XFeat() plain ctor is randomly initialised in kornia 0.9.x.
            # from_pretrained() downloads and loads the real checkpoint.
            model = _KF.XFeat.from_pretrained()
            model.eval()
        elif preset == "ALIKED":
            # KF.ALIKED.from_pretrained confirmed working by user (kornia 0.9.0)
            model = _KF.ALIKED.from_pretrained("aliked-n16")
            model.eval()
        else:
            raise ValueError(f"Unknown learned preset: {preset}")
        _LEARNED_MODELS[preset] = model
    return _LEARNED_MODELS[preset]


def _build_learned_pool(img: np.ndarray, preset: str) -> dict:
    """
    Run a kornia model on a grayscale image and return a pool dict.

    Pool contract (same as OpenCV path):
        {"pts":  float32 (M, 2) — (x, y) in original-image pixels, strongest first
         "desc": float32 (M, D) — matching descriptor for each keypoint}

    Coordinate guarantee: kornia XFeat / ALIKED return coordinates already mapped
    back to input-tensor pixel dimensions, so no rescaling is needed.
    """
    model = _get_learned_model(preset)

    # Grayscale H×W uint8  →  (1, 1, H, W) float32 in [0, 1]
    img_t = _torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    with _torch.no_grad():
        if preset == "XFeat":
            # detectAndCompute → List[Dict]; one dict per batch item.
            # Keys: "keypoints" (N,2), "descriptors" (N,64), "scores" (N,)
            out_list = model.detectAndCompute(img_t, top_k=MAX_POOL)
            r = out_list[0]
            kpts   = r["keypoints"].cpu().numpy().astype(np.float32)     # (N, 2)
            desc   = r["descriptors"].cpu().numpy().astype(np.float32)   # (N, 64)
            scores = r["scores"].cpu().numpy().flatten()                  # (N,)

        elif preset == "ALIKED":
            # forward() returns a list of ALIKEDFeatures objects (one per batch item).
            # Each object has direct tensor attributes — do NOT treat as dict.
            out = model(img_t)
            r = out[0]                                                  # batch size = 1
            kpts   = r.keypoints.cpu().numpy().astype(np.float32)      # (N, 2) (x, y)
            desc   = r.descriptors.cpu().numpy().astype(np.float32)    # (N, D)
            scores = r.keypoint_scores.cpu().numpy().flatten()         # (N,)

        else:
            raise ValueError(f"Unknown learned preset: {preset}")

    if len(kpts) == 0:
        return {"pts": np.zeros((0, 2), np.float32), "desc": None}

    # Sort strongest-first and cap at MAX_POOL (consistent with OpenCV path)
    order = np.argsort(-scores, kind="stable")[:MAX_POOL]
    return {"pts": kpts[order], "desc": desc[order]}


# ---- BEBLID descriptor size constant ------------------------------------
def _beblid_256_bits():
    """Return the BEBLID_SIZE_256_BITS integer constant, tolerating OpenCV naming changes."""
    xf = getattr(cv2, "xfeatures2d", None)
    if xf is None:
        raise AttributeError("cv2.xfeatures2d not found — install opencv-contrib-python")
    # Try the documented constant name first
    val = getattr(xf, "BEBLID_SIZE_256_BITS", None)
    if val is not None:
        return val
    # Fallback: internal integer code used by OpenCV source (value = 101)
    return 101

_BEBLID_NBITS = _beblid_256_bits()

# preset -> (detector_kind, descriptor_kind, descriptor_type)
PRESETS = {
    # ---- original 11 presets (unchanged) --------------------------------
    "ORB":        ("ORB",   "ORB",   "binary"),
    "SIFT":       ("SIFT",  "SIFT",  "float"),
    "AKAZE":      ("AKAZE", "AKAZE", "binary"),
    "BRISK":      ("BRISK", "BRISK", "binary"),
    "FAST+BRIEF": ("FAST",  "BRIEF", "binary"),
    "KAZE":       ("KAZE",  "KAZE",  "float"),
    "GFTT+BRIEF": ("GFTT",  "BRIEF", "binary"),
    "GFTT+SIFT":  ("GFTT",  "SIFT",  "float"),
    "STAR+BRIEF": ("STAR",  "BRIEF", "binary"),
    "ORB+FREAK":  ("ORB",   "FREAK", "binary"),
    "FAST+FREAK": ("FAST",  "FREAK", "binary"),
    # ---- new presets (14-preset setup) ----------------------------------
    "XFeat":      ("XFeat",  "XFeat",  "float"),   # kornia: 64-dim float descriptors
    "ALIKED":     ("ALIKED", "ALIKED", "float"),   # kornia aliked-n16: float descriptors
    "ORB+BEBLID": ("ORB",    "BEBLID", "binary"),  # ORB detector + BEBLID 256-bit
}
PRESET_NAMES = list(PRESETS)
VALID_MATCHERS = {"binary": ["BF"], "float": ["BF", "FLANN"]}


@lru_cache(maxsize=None)
def _detector(kind):
    """Permissive settings: we want a big candidate pool, then cut to top-N."""
    if kind == "ORB":   return make("ORB", nfeatures=MAX_POOL)
    if kind == "SIFT":  return make("SIFT", nfeatures=0, contrastThreshold=0.01)
    if kind == "AKAZE": return make("AKAZE", threshold=1e-4)
    if kind == "BRISK": return make("BRISK", thresh=10)
    if kind == "KAZE":  return make("KAZE", threshold=1e-4)
    if kind == "FAST":  return make("FastFeatureDetector", threshold=5, nonmaxSuppression=True)
    if kind == "GFTT":  return make("GFTTDetector", maxCorners=MAX_POOL, qualityLevel=0.001, minDistance=3)
    if kind == "STAR":  return make("StarDetector", responseThreshold=10)
    raise ValueError(kind)


@lru_cache(maxsize=None)
def _descriptor(kind):
    names = {"ORB": "ORB", "SIFT": "SIFT", "AKAZE": "AKAZE", "BRISK": "BRISK",
             "KAZE": "KAZE", "BRIEF": "BriefDescriptorExtractor", "FREAK": "FREAK"}
    if kind == "BEBLID":
        # scale_factor=1.0 is the OpenCV recommendation for ORB keypoints
        return make("BEBLID", scale_factor=1.0, n_bits=_BEBLID_NBITS)
    return make(names[kind])


# ---- pools: the expensive, N-independent part ---------------------------
def _build_pool(img, preset):
    """
    Detect, describe ALL candidates; keep survivors sorted strongest-first.

    For learned presets (XFeat, ALIKED), delegates to _build_learned_pool.
    For OpenCV presets, uses the existing detector/descriptor pipeline.
    """
    if preset in LEARNED_PRESETS:
        return _build_learned_pool(img, preset)

    # --- OpenCV path (unchanged from original) ---
    det_kind, desc_kind, _ = PRESETS[preset]
    kps = _detector(det_kind).detect(img, None)
    kps = sorted(kps, key=lambda k: -k.response)[:MAX_POOL]
    kps, desc = _descriptor(desc_kind).compute(img, list(kps))
    if desc is None or len(kps) == 0:
        return {"pts": np.zeros((0, 2), np.float32), "desc": None}
    order = np.argsort([-k.response for k in kps], kind="stable")
    pts = np.float32([kps[i].pt for i in order])
    return {"pts": pts, "desc": desc[order]}


_pool_cache = OrderedDict()
DISK_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "pool_cache"

def _disk_path(key, preset):
    seq, idx = key
    safe_preset = preset.replace("+", "_")
    return DISK_CACHE_DIR / safe_preset / f"{seq}__{idx}.npz"

def _load_disk(path):
    d = np.load(path)
    return {"pts": d["pts"], "desc": d["desc"] if d["desc"].size else None}

def _save_disk(path, pool):
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = pool["desc"] if pool["desc"] is not None else np.zeros((0, 0))
    np.savez_compressed(path, pts=pool["pts"], desc=desc)

def get_pool(img, preset, key=None):
    """Returns (pool, build_seconds). build_seconds is 0.0 on a cache hit
    (memory OR disk). key must uniquely identify the image, e.g.
    (sequence_name, image_index). With key=None nothing is cached."""
    if key is None:
        t0 = time.perf_counter()
        pool = _build_pool(img, preset)
        return pool, time.perf_counter() - t0

    ck = (key, preset)
    if ck in _pool_cache:
        _pool_cache.move_to_end(ck)
        return _pool_cache[ck], 0.0

    path = _disk_path(key, preset)
    if path.exists():
        pool = _load_disk(path)
        _pool_cache[ck] = pool
        if len(_pool_cache) > POOL_CACHE_SIZE:
            _pool_cache.popitem(last=False)
        return pool, 0.0

    t0 = time.perf_counter()
    pool = _build_pool(img, preset)
    dt = time.perf_counter() - t0
    _save_disk(path, pool)
    _pool_cache[ck] = pool
    if len(_pool_cache) > POOL_CACHE_SIZE:
        _pool_cache.popitem(last=False)
    return pool, dt


# ---- matching: the cheap, N-dependent part ------------------------------
def match_ratio(d1, d2, preset, matcher, ratio):
    dtype = PRESETS[preset][2]
    if matcher not in VALID_MATCHERS[dtype]:
        raise ValueError(f"matcher {matcher} invalid for {preset} ({dtype})")
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    if dtype == "binary":
        m = cv2.BFMatcher(cv2.NORM_HAMMING)
    elif matcher == "BF":
        m = cv2.BFMatcher(cv2.NORM_L2)
    else:
        m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        d1, d2 = d1.astype(np.float32), d2.astype(np.float32)
    pairs = m.knnMatch(d1, d2, k=2)
    return [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def run_pipeline(img1, img2, preset, n_kp, matcher="BF", ratio=0.75, key1=None, key2=None):
    """Top-n_kp of each cached pool -> match -> ratio test.
    Pass key1/key2 (e.g. (seq, 1) and (seq, k)) to use the cache."""
    p1, b1 = get_pool(img1, preset, key1)
    p2, b2 = get_pool(img2, preset, key2)

    t0 = time.perf_counter()
    n1, n2 = min(n_kp, len(p1["pts"])), min(n_kp, len(p2["pts"]))
    d1 = p1["desc"][:n1] if n1 else None
    d2 = p2["desc"][:n2] if n2 else None
    good = match_ratio(d1, d2, preset, matcher, ratio)
    if good:
        pts1 = np.float32([p1["pts"][m.queryIdx] for m in good])
        pts2 = np.float32([p2["pts"][m.trainIdx] for m in good])
    else:
        pts1 = pts2 = np.zeros((0, 2), np.float32)
    match_s = time.perf_counter() - t0

    return dict(pts1=pts1, pts2=pts2, n_kp1=n1, n_kp2=n2, n_kp=min(n1, n2),
                n_matches=len(good), pool_s=b1 + b2, match_s=match_s,
                time_s=b1 + b2 + match_s,
                pool_n1=len(p1["pts"]), pool_n2=len(p2["pts"]))