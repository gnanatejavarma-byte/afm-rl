import os, glob, cv2, numpy as np

ROOT = "data/megadepth/megadepth_test_1500"
INFO = "data/megadepth/megadepth_test_1500_scene_info"
LONG_SIDE = 1200
_trapz = getattr(np, "trapezoid", None) or np.trapz
_scenes = {}


def _scene(fname):
    if fname not in _scenes:
        _scenes[fname] = np.load(os.path.join(INFO, fname), allow_pickle=True)
    return _scenes[fname]


def list_pairs():
    out = []
    for f in sorted(glob.glob(os.path.join(INFO, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        for (i, j), *_ in d["pair_infos"]:
            out.append((os.path.basename(f), int(i), int(j)))
    return out


def bucket(fname):
    """Overlap range of a pair file, e.g. '0015_0.1_0.3.npz' -> '0.1-0.3'."""
    _, a, b = fname[:-4].split("_")
    return f"{a}-{b}"


def image_key(fname, k):
    """Unique cache key for an image: (scene, file stem)."""
    d = _scene(fname)
    stem = os.path.splitext(os.path.basename(str(d["image_paths"][k])))[0]
    return ("md" + fname.split("_")[0], stem)


def pair_keys(fname, i, j):
    return image_key(fname, i), image_key(fname, j)


def load_image(fname, k):
    d = _scene(fname)
    img = cv2.imread(os.path.join(ROOT, str(d["image_paths"][k])), cv2.IMREAD_GRAYSCALE)
    K = d["intrinsics"][k].astype(np.float64).copy()
    s = LONG_SIDE / max(img.shape)
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        K[:2] *= s
    return img, K


def load_pair(fname, i, j):
    d = _scene(fname)
    img0, K0 = load_image(fname, i)
    img1, K1 = load_image(fname, j)
    T = d["poses"][j] @ np.linalg.inv(d["poses"][i])      # relative pose 0 -> 1
    return img0, img1, K0, K1, T


def estimate_pose_ex(p0, p1, K0, K1, thr=0.5):
    """Essential-matrix RANSAC. Returns (R, t, inlier_ratio) or None."""
    if len(p0) < 5:
        return None
    n0 = (p0 - K0[[0, 1], [2, 2]]) / K0[[0, 1], [0, 1]]
    n1 = (p1 - K1[[0, 1], [2, 2]]) / K1[[0, 1], [0, 1]]
    f = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.RANSAC, prob=0.99999, threshold=thr / f)
    if E is None or mask is None:
        return None
    ratio = float(mask.sum()) / len(p0)
    best, ret = 0, None
    for e in np.split(E, len(E) // 3):
        n, R, t, _ = cv2.recoverPose(e, n0, n1, np.eye(3), 1e9, mask=mask)
        if n > best:
            best, ret = n, (R, t[:, 0], ratio)
    return ret


def estimate_pose(p0, p1, K0, K1, thr=0.5):
    r = estimate_pose_ex(p0, p1, K0, K1, thr)
    return None if r is None else r[:2]


def pose_error(T, R, t):
    """Max of rotation error and translation-direction error, in degrees."""
    c = np.clip((np.trace(R.T @ T[:3, :3]) - 1) / 2, -1, 1)
    r_err = np.rad2deg(np.abs(np.arccos(c)))
    v = np.clip(np.dot(t, T[:3, 3]) / (np.linalg.norm(t) * np.linalg.norm(T[:3, 3]) + 1e-12), -1, 1)
    t_err = np.rad2deg(np.arccos(v))
    return max(r_err, min(t_err, 180 - t_err))


def pose_auc(errors, thresholds=(5, 10, 20)):
    e = np.sort(np.asarray(errors, dtype=float))
    recall = (np.arange(len(e)) + 1) / len(e)
    e, recall = np.r_[0.0, e], np.r_[0.0, recall]
    out = []
    for t in thresholds:
        k = np.searchsorted(e, t)
        out.append(_trapz(np.r_[recall[:k], recall[k - 1]], x=np.r_[e[:k], t]) / t)
    return out