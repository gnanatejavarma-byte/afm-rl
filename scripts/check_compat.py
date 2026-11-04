"""
check_compat.py  —  sanity checks for the 14-preset setup.
Run from the project root:
    python scripts/check_compat.py
All sections should print OK.  Any FAIL line must be fixed before training.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from cv_compat import make

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "\033[92mOK  \033[0m"
FAIL = "\033[91mFAIL\033[0m"

def _load_real_image():
    """Return a real HPatches grayscale image (first val sequence, img 1)."""
    from hpatches import load_split, load_pair
    val = load_split()["val"]
    seq = val[0]
    img1, _, _ = load_pair(seq, 2)   # img1 is the reference image
    print(f"  Using HPatches image: {seq}/1.ppm  shape={img1.shape}")
    return img1, seq

# ---- 1. OpenCV classic detectors / descriptors ----------------------------
print("=== 1. OpenCV classic algorithms ===")
cv_names = ["ORB", "SIFT", "AKAZE", "BRISK", "KAZE", "FastFeatureDetector",
            "GFTTDetector", "StarDetector", "BriefDescriptorExtractor", "FREAK"]
for name in cv_names:
    try:
        obj = make(name)
        print(f"{PASS}  {name} -> {type(obj).__name__}")
    except Exception as e:
        print(f"{FAIL}  {name}: {e}")

# ---- 2. BEBLID (256-bit) --------------------------------------------------
print("\n=== 2. BEBLID descriptor (256-bit, for ORB+BEBLID preset) ===")
try:
    xf = getattr(cv2, "xfeatures2d", None)
    if xf is None:
        raise AttributeError("cv2.xfeatures2d missing — install opencv-contrib-python")
    c256 = getattr(xf, "BEBLID_SIZE_256_BITS", 101)
    c512 = getattr(xf, "BEBLID_SIZE_512_BITS", 100)
    print(f"  BEBLID_SIZE_256_BITS constant = {c256}  (expect 101)")
    print(f"  BEBLID_SIZE_512_BITS constant = {c512}  (expect 100)")
    beblid = make("BEBLID", scale_factor=1.0, n_bits=c256)
    print(f"{PASS}  BEBLID (256-bit) -> {type(beblid).__name__}")
except Exception as e:
    print(f"{FAIL}  BEBLID: {e}")

# ---- 3. XFeat (must use from_pretrained) ----------------------------------
print("\n=== 3. kornia XFeat (from_pretrained) ===")
try:
    import torch
    import kornia.feature as KF
    print(f"  kornia imported OK | torch={torch.__version__}")

    img_real, seq_name = _load_real_image()
    H, W = img_real.shape[:2]
    img_t = torch.from_numpy(img_real.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)

    model_xf = KF.XFeat.from_pretrained()   # NOT KF.XFeat() — that is randomly initialised
    model_xf.eval()

    with torch.no_grad():
        out = model_xf.detectAndCompute(img_t, top_k=4096)

    r = out[0]   # list of dicts, one per batch item
    kpts  = r["keypoints"].cpu().numpy()    # (N, 2)
    desc  = r["descriptors"].cpu().numpy()  # (N, 64)
    scores = r["scores"].cpu().numpy()      # (N,)

    n = len(kpts)
    assert n > 100, f"Expected > 100 keypoints but got {n} — weights may not be loaded"
    in_bounds = (kpts[:, 0].min() >= 0 and kpts[:, 0].max() <= W and
                 kpts[:, 1].min() >= 0 and kpts[:, 1].max() <= H)
    assert in_bounds, f"Keypoints out of image bounds [0,{W}]x[0,{H}]"

    print(f"  keypoints : {kpts.shape}  (x in [{kpts[:,0].min():.1f}, {kpts[:,0].max():.1f}], "
          f"y in [{kpts[:,1].min():.1f}, {kpts[:,1].max():.1f}])")
    print(f"  descriptors: {desc.shape}")
    print(f"  scores     : min={scores.min():.4f}  max={scores.max():.4f}")
    print(f"  desc stats : mean={np.abs(desc).mean():.4f}  std={desc.std():.4f}")
    print(f"{PASS}  XFeat — {n} keypoints, desc dim={desc.shape[1]}, coords in bounds")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"{FAIL}  XFeat: {e}")

# ---- 4. ALIKED (aliked-n16) -----------------------------------------------
print("\n=== 4. kornia ALIKED (aliked-n16, from_pretrained) ===")
try:
    import torch
    import kornia.feature as KF

    # Reuse the same real image loaded above (or load fresh if section 3 failed)
    try:
        _ = img_real
    except NameError:
        img_real, seq_name = _load_real_image()
        H, W = img_real.shape[:2]
    img_t = torch.from_numpy(img_real.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)

    model_al = KF.ALIKED.from_pretrained("aliked-n16")
    model_al.eval()

    with torch.no_grad():
        out = model_al(img_t)

    # out is a list of ALIKEDFeatures objects — access attributes directly
    assert isinstance(out, (list, tuple)), f"Expected list, got {type(out)}"
    r = out[0]   # ALIKEDFeatures for the first (and only) batch item
    print(f"  ALIKEDFeatures attrs: {[a for a in dir(r) if not a.startswith('_')]}")

    kpts   = r.keypoints.cpu().numpy().astype(np.float32)       # (N, 2)
    desc   = r.descriptors.cpu().numpy().astype(np.float32)     # (N, D)
    scores = r.keypoint_scores.cpu().numpy().flatten()           # (N,)

    n = len(kpts)
    D = desc.shape[1]
    in_bounds = (kpts[:, 0].min() >= 0 and kpts[:, 0].max() <= W and
                 kpts[:, 1].min() >= 0 and kpts[:, 1].max() <= H)

    print(f"  keypoints : {kpts.shape}  (x in [{kpts[:,0].min():.1f}, {kpts[:,0].max():.1f}], "
          f"y in [{kpts[:,1].min():.1f}, {kpts[:,1].max():.1f}])")
    print(f"  descriptors: {desc.shape}")
    print(f"  scores     : min={scores.min():.4f}  max={scores.max():.4f}")
    print(f"  coords in bounds [{W}x{H}]: {in_bounds}")
    print(f"{PASS}  ALIKED — {n} keypoints, desc dim={D}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"{FAIL}  ALIKED: {e}")

# ---- 5. ORB+BEBLID end-to-end (detect + describe on real image) ----------
print("\n=== 5. ORB+BEBLID end-to-end (real image) ===")
try:
    try:
        _ = img_real
    except NameError:
        img_real, _ = _load_real_image()

    orb  = make("ORB", nfeatures=5000)
    xf2  = getattr(cv2, "xfeatures2d")
    c256 = getattr(xf2, "BEBLID_SIZE_256_BITS", 101)
    beblid = make("BEBLID", scale_factor=1.0, n_bits=c256)

    kps  = orb.detect(img_real, None)
    kps, desc = beblid.compute(img_real, kps)
    print(f"  ORB detected {len(kps)} keypoints")
    print(f"  BEBLID desc shape: {desc.shape}  dtype={desc.dtype}  "
          f"(expect (N,32), uint8)")
    assert desc.dtype == np.uint8, "BEBLID descriptor should be uint8 (binary)"
    assert desc.shape[1] == 32, f"256-bit = 32 bytes, got {desc.shape[1]}"
    print(f"{PASS}  ORB+BEBLID — {len(kps)} keypoints, desc 256-bit")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"{FAIL}  ORB+BEBLID: {e}")

# ---- 6. OBS_DIM = 34 -------------------------------------------------------
print("\n=== 6. Environment observation dimension ===")
try:
    from level2_env import OBS_DIM
    import pipeline_utils as pu
    print(f"  N_PRESETS  = {len(pu.PRESET_NAMES)}  (expect 14)")
    print(f"  PRESET_NAMES = {pu.PRESET_NAMES}")
    print(f"  OBS_DIM    = {OBS_DIM}  (expect 34)")
    assert len(pu.PRESET_NAMES) == 14, f"Expected 14, got {len(pu.PRESET_NAMES)}"
    assert OBS_DIM == 34, f"Expected 34, got {OBS_DIM}"
    print(f"{PASS}  OBS_DIM=34 and N_PRESETS=14 confirmed")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"{FAIL}  OBS_DIM check: {e}")