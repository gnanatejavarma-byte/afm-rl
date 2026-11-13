import cv2
import numpy as np
import hashlib

REPROJ_PX = 3.0    # a match is "correct" if it lands within this many pixels of GT
CORNER_PX = 3.0    # homography counts as accurate if mean corner error is below this
REWARD_FALLOFF = 5.0  # controls how fast reward_acc decays with corner error


def get_ransac_seed(seq, k, method="default"):
    """Deterministic 31-bit integer seed derived from (seq, k, method)."""
    s = f"{seq}_{k}_{method}"
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def _project(H, pts):
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64)).reshape(-1, 2)


def reward_accuracy(corner_err, falloff=REWARD_FALLOFF):
    """Candidate D: smooth score in (0, 1]. 1.0 = perfect homography match,
    approaches 0 as corner error grows or when no homography could be found
    (corner_err = inf, from a failed/absent RANSAC fit)."""
    return float(np.exp(-corner_err / falloff))


def evaluate_matches(pts1, pts2, H_gt, img_shape, ransac_seed=None):
    """pts1: matched points in image 1, pts2: their partners in image k,
    H_gt: ground-truth homography image1 -> image k."""
    n = len(pts1)
    out = dict(n_matches=n, n_correct=0, precision=0.0,
               ransac_ratio=0.0, corner_err=float("inf"), h_ok=0.0,
               reward_acc=0.0)
    if n == 0:
        return out

    # 1) ground-truth check: where SHOULD each image-1 point land?
    err = np.linalg.norm(_project(H_gt, pts1) - pts2, axis=1)
    out["n_correct"] = int((err < REPROJ_PX).sum())
    out["precision"] = out["n_correct"] / n

    # 2) RANSAC (needs >= 4 matches) + corner error of the estimated homography
    if n >= 4:
        if ransac_seed is not None:
            cv2.setRNGSeed(int(ransac_seed) & 0x7FFFFFFF)
        H_est, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, REPROJ_PX)
        if H_est is not None and mask is not None:
            out["ransac_ratio"] = float(mask.sum()) / n
            h, w = img_shape[:2]
            corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
            with np.errstate(all="ignore"):
                ce = np.linalg.norm(_project(H_gt, corners) - _project(H_est, corners), axis=1).mean()
            if np.isfinite(ce):
                out["corner_err"] = float(ce)
                out["h_ok"] = float(ce < CORNER_PX)

    out["reward_acc"] = reward_accuracy(out["corner_err"])
    return out


def compute_label_free_proxy(pts1, pts2, img_shape=None, ransac_seed=None):
    """
    Computes a 100% label-free proxy score in [0, 1] without ground-truth homography.
    Uses RANSAC inlier count, inlier ratio, and inlier reprojection error.
    """
    n = len(pts1)
    out = dict(
        n_matches=n,
        n_inliers=0,
        inlier_ratio=0.0,
        mean_reproj=float("inf"),
        proxy=0.0,
        H_est=None,
    )
    if n < 4:
        return out

    if ransac_seed is not None:
        cv2.setRNGSeed(int(ransac_seed) & 0x7FFFFFFF)
    H_est, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, REPROJ_PX)
    if H_est is None or mask is None:
        return out

    inl_mask = (mask.ravel() == 1)
    n_inliers = int(inl_mask.sum())
    inlier_ratio = n_inliers / n if n > 0 else 0.0
    out["n_inliers"] = n_inliers
    out["inlier_ratio"] = inlier_ratio
    out["H_est"] = H_est

    if n_inliers >= 4:
        p1_inl = pts1[inl_mask].reshape(-1, 1, 2)
        p2_proj = cv2.perspectiveTransform(p1_inl, H_est).reshape(-1, 2)
        reproj_errs = np.linalg.norm(p2_proj - pts2[inl_mask], axis=1)
        mean_reproj = float(np.mean(reproj_errs))
        out["mean_reproj"] = mean_reproj

        # Proxy formula: count saturation (up to 120 inliers) * inlier ratio * reproj factor
        count_sat = min(1.0, n_inliers / 120.0)
        reproj_factor = float(np.exp(-mean_reproj / 2.0))
        proxy = count_sat * (0.6 * inlier_ratio + 0.4 * reproj_factor)
        out["proxy"] = float(np.clip(proxy, 0.0, 1.0))

    return out
