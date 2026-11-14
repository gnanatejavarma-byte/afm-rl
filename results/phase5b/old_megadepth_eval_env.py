import os
import hashlib
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from stable_baselines3 import PPO

import megadepth as md
import pipeline_utils as pu
from image_stats import compute_pair_stats, normalize_stats, N_STATS

POOL_CACHE_DIR = Path("data/pool_cache_megadepth")
STATS_CACHE_DIR = Path("data/megadepth/stats_cache")

OBS_DIM = 34
MAX_STEPS = 5
MATCH_NORM = 200.0


def compute_epipolar_proxy(pts1, pts2, K0, K1, thr=0.5):
    """
    Computes a 100% label-free epipolar proxy score in [0, 1] using Essential matrix RANSAC.
    Follows Eq. 7 functional form:
        proxy = count_sat * (0.6 * inlier_ratio + 0.4 * reproj_factor)
    using Sampson epipolar distance in pixels.
    Uses camera intrinsics K0, K1. Never touches ground-truth pose.
    """
    n = len(pts1)
    out = dict(
        n_matches=n,
        n_inliers=0,
        inlier_ratio=0.0,
        mean_sampson=float("inf"),
        proxy=0.0,
        E_est=None,
        mask=None,
    )
    if n < 5:
        return out

    # Normalized ray coordinates
    n0 = (pts1.astype(np.float64) - K0[[0, 1], [2, 2]]) / K0[[0, 1], [0, 1]]
    n1 = (pts2.astype(np.float64) - K1[[0, 1], [2, 2]]) / K1[[0, 1], [0, 1]]
    f_mean = float(np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]]))

    # Essential matrix RANSAC
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.RANSAC, prob=0.99999, threshold=thr / f_mean)
    if E is None or mask is None:
        return out

    inl_mask = (mask.ravel() == 1)
    n_inliers = int(inl_mask.sum())
    inlier_ratio = n_inliers / n if n > 0 else 0.0
    out["n_inliers"] = n_inliers
    out["inlier_ratio"] = inlier_ratio
    out["E_est"] = E
    out["mask"] = mask

    if n_inliers >= 5:
        # Pick the essential matrix block if multiple returned
        E_mat = E[:3, :3] if len(E) >= 3 else E
        
        # Fundamental matrix: F = K1^(-T) @ E @ K0^(-1)
        K0_inv = np.linalg.inv(K0)
        K1_inv_T = np.linalg.inv(K1).T
        F = K1_inv_T @ E_mat @ K0_inv

        # Compute Sampson error in pixels across inliers
        p1_h = np.column_stack([pts1[inl_mask], np.ones(n_inliers)])
        p2_h = np.column_stack([pts2[inl_mask], np.ones(n_inliers)])

        Fx1 = (F @ p1_h.T).T
        Ftx2 = (F.T @ p2_h.T).T
        x2tFx1 = np.sum(p2_h * Fx1, axis=1)

        denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
        sampson_sq = (x2tFx1**2) / np.maximum(denom, 1e-12)
        mean_sampson = float(np.mean(np.sqrt(np.maximum(sampson_sq, 0.0))))
        out["mean_sampson"] = mean_sampson

        # Eq. 7 formula with frozen constants
        count_sat = min(1.0, n_inliers / 120.0)
        reproj_factor = float(np.exp(-mean_sampson / 2.0))
        proxy = count_sat * (0.6 * inlier_ratio + 0.4 * reproj_factor)
        out["proxy"] = float(np.clip(proxy, 0.0, 1.0))

    return out


def get_megadepth_pool(img, preset, image_path, cache_dir=POOL_CACHE_DIR):
    """
    Returns the keypoint pool (pts, desc) at N=4000 for a MegaDepth image.
    Caches pool on disk under cache_dir.
    Key includes: image relative path hash, preset name, and extractor parameters.
    """
    cache_dir = Path(cache_dir) / preset
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Param string for exact versioning
    param_str = f"nfeatures=4000_preset={preset}"
    key_str = f"{image_path}_{preset}_{param_str}"
    h = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{h}.npz"

    if cache_file.exists():
        data = np.load(cache_file)
        desc = data["desc"] if data["desc"].size else None
        return {"pts": data["pts"], "desc": desc}

    # Extract fresh at N=4000
    p = pu._build_pool(img, preset)
    pts = p["pts"][:4000]
    desc = p["desc"][:4000] if p["desc"] is not None else None

    if desc is None:
        desc = np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(cache_file, pts=pts, desc=desc)
    return {"pts": pts, "desc": desc if desc.size else None}


def run_pipeline_megadepth(img1, img2, preset, n_kp, path1, path2, matcher="BF", ratio=0.75):
    """
    Runs feature matching for a MegaDepth pair using the per-image pool cache.
    Top-n_kp sliced from cached N=4000 pool.
    """
    p1 = get_megadepth_pool(img1, preset, path1)
    p2 = get_megadepth_pool(img2, preset, path2)

    n1 = min(n_kp, len(p1["pts"]))
    n2 = min(n_kp, len(p2["pts"]))
    d1 = p1["desc"][:n1] if n1 else None
    d2 = p2["desc"][:n2] if n2 else None

    good = pu.match_ratio(d1, d2, preset, matcher, ratio)
    if good:
        pts1 = np.float32([p1["pts"][m.queryIdx] for m in good])
        pts2 = np.float32([p2["pts"][m.trainIdx] for m in good])
    else:
        pts1 = pts2 = np.zeros((0, 2), np.float32)

    return {
        "pts1": pts1,
        "pts2": pts2,
        "n_kp1": n1,
        "n_kp2": n2,
        "n_kp": min(n1, n2),
        "n_matches": len(good),
        "pool_n1": len(p1["pts"]),
        "pool_n2": len(p2["pts"]),
    }


class MegaDepthEvalEnv(gym.Env):
    """
    Label-free evaluation environment for MegaDepth-1500.
    Operates strictly in obs_mode="proxy".
    Ground-truth pose is protected by a raise-on-access guard during rollouts.
    """
    metadata = {"render_modes": []}

    def __init__(self, lam=0.1, eval_mode=True):
        super().__init__()
        self.lam = lam
        self.eval_mode = eval_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)

        self._in_eval_rollout = False
        self._T_gt = None
        self._stats_cache = {}

    @property
    def T(self):
        """Guard property: strictly forbids access to GT pose during eval rollouts."""
        if getattr(self, "_in_eval_rollout", False):
            raise RuntimeError("CRITICAL LEAK DETECTED: Ground-truth pose self.T was accessed during an eval_mode rollout!")
        return self._T_gt

    @T.setter
    def T(self, value):
        self._T_gt = value

    def _stats(self, fname, i, j, img1, img2):
        key = (fname, i, j)
        if key not in self._stats_cache:
            path = STATS_CACHE_DIR / f"{fname}__{i}__{j}.npy"
            if path.exists():
                self._stats_cache[key] = np.load(path)
            else:
                s = normalize_stats(compute_pair_stats(img1, img2))
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, s)
                self._stats_cache[key] = s
        return self._stats_cache[key]

    def reset_to(self, fname, i, j, preset):
        """Initializes pair and preset for rollout."""
        self._in_eval_rollout = False
        self.fname = fname
        self.i = i
        self.j = j
        self.img1, self.K0 = md.load_image(fname, i)
        self.img2, self.K1 = md.load_image(fname, j)
        
        # Load GT pose safely into protected variable
        d = md._scene(fname)
        self._T_gt = d["poses"][j] @ np.linalg.inv(d["poses"][i])
        
        self.path1 = str(d["image_paths"][i])
        self.path2 = str(d["image_paths"][j])
        
        self.preset = preset
        self.preset_idx = pu.PRESET_NAMES.index(preset)
        self.n_kp = pu.N_START
        self.steps = 0
        self.cum_kp = 0

        self.pair_stats = self._stats(fname, i, j, self.img1, self.img2)
        self.prev_proxy = 0.0
        self.prev_inlier_ratio = 0.0
        self.prev_n_matches = 0.0

        # Enable rollout guard
        self._in_eval_rollout = True
        self._run_and_eval()
        self.cum_kp += self.effective_kp

        # Scoreboard tracking (proxy utility)
        self.best_proxy_utility = self.proxy_utility
        self.best_proxy = self.last_proxy
        self.best_n_kp = self.n_kp
        self.best_n_matches = self.last_r["n_matches"]
        self.best_pts1 = self.last_r["pts1"]
        self.best_pts2 = self.last_r["pts2"]

        return self._get_obs(), self._get_info()

    def _run_and_eval(self):
        """Runs pipeline and computes label-free epipolar proxy."""
        r = run_pipeline_megadepth(
            self.img1, self.img2, self.preset, self.n_kp,
            self.path1, self.path2, matcher="BF", ratio=0.75
        )
        self.effective_kp = r["n_kp"]
        self.pool_size = min(r["pool_n1"], r["pool_n2"])
        self.last_r = r

        proxy_res = compute_epipolar_proxy(r["pts1"], r["pts2"], self.K0, self.K1)
        self.last_proxy_res = proxy_res
        self.last_proxy = proxy_res["proxy"]
        self.last_inlier_ratio = proxy_res["inlier_ratio"]
        self.last_n_matches = min(r["n_matches"], MATCH_NORM) / MATCH_NORM

        # Proxy utility: proxy - lambda * (n_kp / N_MAX)
        self.proxy_utility = self.last_proxy - self.lam * (self.effective_kp / pu.N_MAX)
        return proxy_res

    def _get_obs(self):
        one_hot = np.zeros(len(pu.PRESET_NAMES), dtype=np.float32)
        one_hot[self.preset_idx] = 1.0

        budget_norm = float(self.n_kp / pu.N_MAX)
        effective_norm = float(self.effective_kp / pu.N_MAX)
        pool_norm = float(self.pool_size / pu.N_MAX)
        step_norm = float(self.steps / MAX_STEPS)

        delta_proxy = self.last_proxy - self.prev_proxy
        delta_ratio = self.last_inlier_ratio - self.prev_inlier_ratio
        delta_matches = self.last_n_matches - self.prev_n_matches

        obs = np.concatenate([
            self.pair_stats,
            one_hot,
            np.array([
                budget_norm,
                effective_norm,
                pool_norm,
                step_norm,
                self.last_proxy,
                delta_proxy,
                self.last_inlier_ratio,
                delta_ratio,
                self.last_n_matches,
                delta_matches,
                0.0,  # placeholder
            ], dtype=np.float32),
        ])
        return obs[:OBS_DIM].astype(np.float32)

    def _get_info(self):
        return {
            "fname": self.fname, "i": self.i, "j": self.j,
            "preset": self.preset, "n_kp": self.n_kp, "steps": self.steps,
            "effective_kp": self.effective_kp, "cum_kp": self.cum_kp,
            "last_proxy": self.last_proxy,
            "best_proxy": getattr(self, "best_proxy", self.last_proxy),
            "best_n_kp": getattr(self, "best_n_kp", self.n_kp),
            "proxy_utility": self.proxy_utility,
        }

    def step(self, action):
        """
        Actions:
            0: STOP
            1: n_kp + 500
            2: n_kp - 500
            3: n_kp + 1000
        """
        if not self._in_eval_rollout:
            raise RuntimeError("step called outside active episode rollout!")

        self.steps += 1
        self.prev_proxy = self.last_proxy
        self.prev_inlier_ratio = self.last_inlier_ratio
        self.prev_n_matches = self.last_n_matches

        if action == 0:
            terminated = True
        elif action == 1:
            self.n_kp = min(pu.N_MAX, self.n_kp + 500)
            terminated = False
        elif action == 2:
            self.n_kp = max(pu.N_MIN, self.n_kp - 500)
            terminated = False
        elif action == 3:
            self.n_kp = min(pu.N_MAX, self.n_kp + 1000)
            terminated = False
        else:
            raise ValueError(f"Invalid action {action}")

        truncated = (self.steps >= MAX_STEPS)

        if not terminated:
            self._run_and_eval()
            self.cum_kp += self.effective_kp
            if self.proxy_utility > self.best_proxy_utility:
                self.best_proxy_utility = self.proxy_utility
                self.best_proxy = self.last_proxy
                self.best_n_kp = self.n_kp
                self.best_n_matches = self.last_r["n_matches"]
                self.best_pts1 = self.last_r["pts1"]
                self.best_pts2 = self.last_r["pts2"]

        reward = 0.0  # Eval mode does not compute training reward

        if terminated or truncated:
            # Lift guard post-hoc
            self._in_eval_rollout = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def evaluate_post_hoc(self, mode="final"):
        """
        Post-hoc evaluation with ground truth after rollout completes.
        mode: 'final' (strict STOP state) or 'best' (proxy scoreboard).
        """
        if self._in_eval_rollout:
            raise RuntimeError("evaluate_post_hoc called while rollout is still active!")

        if mode == "final":
            pts1 = self.last_r["pts1"]
            pts2 = self.last_r["pts2"]
            n_kp = self.n_kp
            proxy = self.last_proxy
        elif mode == "best":
            pts1 = self.best_pts1
            pts2 = self.best_pts2
            n_kp = self.best_n_kp
            proxy = self.best_proxy
        else:
            raise ValueError(mode)

        err, ratio = float("inf"), 0.0
        if len(pts1) >= 5:
            est = md.estimate_pose_ex(pts1.astype(np.float64), pts2.astype(np.float64), self.K0, self.K1)
            if est is not None:
                err = float(md.pose_error(self._T_gt, est[0], est[1]))
                ratio = est[2]

        return {
            "fname": self.fname, "i": self.i, "j": self.j,
            "preset": self.preset, "mode": mode, "n_kp": n_kp,
            "cum_kp": self.cum_kp, "steps": self.steps,
            "proxy": proxy, "pose_err": err, "ratio": ratio,
            "succ_5deg": float(err < 5.0),
            "succ_10deg": float(err < 10.0),
            "succ_20deg": float(err < 20.0),
        }
