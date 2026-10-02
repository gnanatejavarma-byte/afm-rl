import cv2
import numpy as np

REPROJ_PX = 3.0    # a match is "correct" if it lands within this many pixels of GT
CORNER_PX = 3.0    # homography counts as accurate if mean corner error is below this
REWARD_FALLOFF = 5.0  # controls how fast reward_acc decays with corner error


def _project(H, pts):
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64)).reshape(-1, 2)


def reward_accuracy(corner_err, falloff=REWARD_FALLOFF):
    """Candidate D: smooth score in (0, 1]. 1.0 = perfect homography match,
    approaches 0 as corner error grows or when no homography could be found
    (corner_err = inf, from a failed/absent RANSAC fit)."""
    return float(np.exp(-corner_err / falloff))


def evaluate_matches(pts1, pts2, H_gt, img_shape):
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