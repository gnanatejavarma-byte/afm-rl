"""Relative Camera Pose Estimation & Evaluation Metrics for MegaDepth.

Computes Essential Matrix, recovers relative rotation and translation (R, t),
calculates angular rotation/translation errors, pose error AUC@5, 10, 20 deg,
and success rates.
"""

from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np


def normalize_points(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Converts 2D pixel coordinates (N, 2) to normalized camera coordinates (N, 2)."""
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    K_inv = np.linalg.inv(K)
    pts_norm = (K_inv @ pts_h.T).T
    return pts_norm[:, :2]


def estimate_relative_pose(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    threshold_px: float = 1.5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, float]:
    """Estimates relative pose (R_est, t_est) using 5-point RANSAC Essential Matrix.

    Returns:
        (R_est, t_est, inlier_count, inlier_ratio)
    """
    n = len(pts1)
    if n < 5:
        return None, None, 0, 0.0

    # 1. Normalize 2D points by their respective camera intrinsics
    pts1_norm = normalize_points(pts1, K1)
    pts2_norm = normalize_points(pts2, K2)

    # 2. Approximate focal length for normalized pixel threshold
    f_mean = 0.5 * ((K1[0, 0] + K1[1, 1]) / 2.0 + (K2[0, 0] + K2[1, 1]) / 2.0)
    thresh_norm = threshold_px / max(f_mean, 1e-6)

    # 3. Estimate Essential Matrix via RANSAC
    E, mask = cv2.findEssentialMat(
        pts1_norm,
        pts2_norm,
        cameraMatrix=np.eye(3, dtype=np.float64),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=thresh_norm,
    )

    if E is None or mask is None:
        return None, None, 0, 0.0

    inliers = int(mask.sum())
    inlier_ratio = float(inliers / n)

    if inliers < 5:
        return None, None, inliers, inlier_ratio

    # Handle multiple candidate solutions if returned
    if E.shape[0] > 3:
        E = E[:3, :3]

    # 4. Recover relative rotation and translation
    try:
        _, R_est, t_est, mask_pose = cv2.recoverPose(
            E,
            pts1_norm,
            pts2_norm,
            cameraMatrix=np.eye(3, dtype=np.float64),
            mask=mask,
        )
        return R_est, t_est, inliers, inlier_ratio
    except Exception:
        return None, None, inliers, inlier_ratio


def compute_pose_errors(
    R_est: Optional[np.ndarray],
    t_est: Optional[np.ndarray],
    R_gt: np.ndarray,
    T_gt: np.ndarray,
) -> Tuple[float, float, float]:
    """Computes angular rotation error, translation error, and combined max error in degrees.

    Returns:
        (err_R_deg, err_t_deg, err_pose_deg)
    """
    if R_est is None or t_est is None:
        return float("inf"), float("inf"), float("inf")

    # Rotation angular error (geodesic distance in SO(3))
    R_diff = R_est @ R_gt.T
    cos_rot = (np.trace(R_diff) - 1.0) / 2.0
    cos_rot = np.clip(cos_rot, -1.0, 1.0)
    err_R = float(np.degrees(np.arccos(cos_rot)))

    # Translation angular error (direction angle)
    t1 = t_est.ravel() / (np.linalg.norm(t_est) + 1e-12)
    t2 = T_gt.ravel() / (np.linalg.norm(T_gt) + 1e-12)
    dot = float(np.dot(t1, t2))
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    # Account for sign ambiguity in epipolar translation baseline
    err_t = float(min(angle, 180.0 - angle))

    err_pose = float(max(err_R, err_t))
    return err_R, err_t, err_pose


def evaluate_pose_from_matches(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    R_gt: np.ndarray,
    T_gt: np.ndarray,
    threshold_px: float = 1.5,
) -> Dict[str, Any]:
    """Complete evaluation pipeline for a set of matches."""
    n_matches = len(pts1)
    if n_matches < 5:
        return {
            "n_matches": n_matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "err_R": float("inf"),
            "err_t": float("inf"),
            "err_pose": float("inf"),
            "success_5": False,
            "success_10": False,
            "success_20": False,
        }

    R_est, t_est, inliers, inlier_ratio = estimate_relative_pose(
        pts1, pts2, K1, K2, threshold_px=threshold_px
    )
    err_R, err_t, err_pose = compute_pose_errors(R_est, t_est, R_gt, T_gt)

    return {
        "n_matches": n_matches,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "err_R": err_R,
        "err_t": err_t,
        "err_pose": err_pose,
        "success_5": bool(err_pose < 5.0),
        "success_10": bool(err_pose < 10.0),
        "success_20": bool(err_pose < 20.0),
    }


def compute_auc(errors: List[float], thresholds: List[float] = [5.0, 10.0, 20.0]) -> Dict[str, float]:
    """Computes standard pose error AUC@5, 10, 20 degrees via numerical trapezoidal integration."""
    err_arr = np.asarray(errors, dtype=np.float64)
    clean_err = np.sort(np.nan_to_num(err_arr, nan=1e9, posinf=1e9))
    n_samples = len(clean_err)
    if n_samples == 0:
        return {f"auc@{int(t)}": 0.0 for t in thresholds}

    # Use np.trapezoid (NumPy 2.x) with fallback to np.trapz (NumPy 1.x)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

    aucs = {}
    for t in thresholds:
        bins = np.linspace(0, t, 100)
        recall = np.searchsorted(clean_err, bins, side="right") / n_samples
        auc_val = float(trapz_fn(recall, bins) / t) * 100.0
        aucs[f"auc@{int(t)}"] = round(auc_val, 2)
    return aucs



def compute_success_rates(errors: List[float], thresholds: List[float] = [5.0, 10.0, 20.0]) -> Dict[str, float]:
    """Computes success percentage (error < threshold) for given thresholds."""
    err_arr = np.asarray(errors, dtype=np.float64)
    n_samples = len(err_arr)
    if n_samples == 0:
        return {f"prec@{int(t)}": 0.0 for t in thresholds}

    rates = {}
    for t in thresholds:
        succ = (err_arr < t).sum()
        rates[f"prec@{int(t)}"] = round(float(succ / n_samples) * 100.0, 2)
    return rates
