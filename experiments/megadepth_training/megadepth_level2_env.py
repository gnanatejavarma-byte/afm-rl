"""MegaDepth-1500 Level-2 Gymnasium Environment.

Implements the sequential keypoint-budget decision process for MegaDepth image pairs.
Mirroring Level-2 HPatches env design, but using MegaDepth-1500 relative pose geometry
and F-RANSAC Sampson surrogate accuracy.

Actions:
  0: +500 keypoints
  1: +1000 keypoints
  2: -500 keypoints
  3: STOP

Observation (31-dim):
  [0..8]   Normalized visual pair statistics (9-dim)
  [9..19]  One-hot encoding of current preset (11-dim)
  [20]     n_kp / N_MAX
  [21]     effective_kp / N_MAX
  [22]     pool_size / N_MAX
  [23]     last_acc (exp(-mean_sampson_error / 5.0))
  [24]     last_inlier_ratio (F-RANSAC inlier ratio)
  [25]     min(n_matches, 1000) / 1000
  [26]     step_fraction (steps / MAX_STEPS)
  [27]     delta_acc (acc_t - acc_{t-1})
  [28]     delta_utility (utility_t - utility_{t-1})
  [29]     is_plateau (1.0 if effective_kp >= pool_size else 0.0)
  [30]     gap_from_best (utility_t - best_utility_seen <= 0)
"""

import random
from pathlib import Path
import sys
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
MEGADEPTH_EXP = PROJECT_ROOT / "experiments" / "megadepth"
EXP_DIR = Path(__file__).resolve().parent

for p in [str(SRC_DIR), str(MEGADEPTH_EXP), str(EXP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline_utils as pu
from image_stats import compute_pair_stats, normalize_stats, N_STATS
from dataset import MegaDepthDataset
from pose_metrics import evaluate_pose_from_matches

MATCHER = "BF"
RATIO = 0.75
STEP_SIZES = [500, 1000, -500, None]  # None = STOP
ACTION_LABELS = ["+500", "+1000", "-500", "STOP"]
MAX_STEPS = 15
DEFAULT_LAMBDA = 0.1
STEP_PENALTY = 0.01
MATCH_NORM = 1000.0

N_PRESETS = len(pu.PRESET_NAMES)
OBS_DIM = N_STATS + N_PRESETS + 11  # 9 + 11 + 11 = 31


def _surrogate_acc(pts1: np.ndarray, pts2: np.ndarray, falloff: float = 5.0) -> tuple[float, float]:
    """Surrogate for reward_acc using F-RANSAC Sampson epipolar distance."""
    n = len(pts1)
    if n < 8:
        return 0.0, 0.0

    try:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.0, 0.999)
        if F is None or mask is None:
            return 0.0, 0.0

        inlier_ratio = float(mask.sum()) / n
        inliers1 = pts1[mask.ravel().astype(bool)]
        inliers2 = pts2[mask.ravel().astype(bool)]

        if len(inliers1) < 1:
            return 0.0, inlier_ratio

        pts1_h = np.hstack([inliers1, np.ones((len(inliers1), 1), dtype=np.float64)])
        pts2_h = np.hstack([inliers2, np.ones((len(inliers2), 1), dtype=np.float64)])
        Fx1 = (F @ pts1_h.T).T
        Ftx2 = (F.T @ pts2_h.T).T
        numer = np.sum(pts2_h * Fx1, axis=1) ** 2
        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-9
        sampson = np.sqrt(np.clip(numer / denom, 0, None))
        mean_sampson = float(sampson.mean())

        surrogate = float(np.exp(-mean_sampson / falloff))
        return surrogate, inlier_ratio
    except Exception:
        return 0.0, 0.0


class MegaDepthLevel2Env(gym.Env):
    """Level-2 RL Environment for MegaDepth-1500."""
    metadata = {"render_modes": []}

    def __init__(self, split: str = "train", lam: float = DEFAULT_LAMBDA, seed: int | None = None):
        super().__init__()
        self.split = split
        self.lam = lam
        self.rng = random.Random(seed)

        # Load dataset and split manifest
        self.dataset = MegaDepthDataset()
        manifest_path = EXP_DIR / "split_manifest.csv"
        if not manifest_path.exists():
            from generate_split import generate_split_manifest
            generate_split_manifest()

        manifest_df = pd.read_csv(manifest_path)
        split_df = manifest_df[manifest_df["split"] == split]
        self.pair_indices = split_df["pair_idx"].tolist()
        assert len(self.pair_indices) > 0, f"No pairs found for split '{split}'"

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)

        # Cache pair visual stats to avoid recomputation
        self._stats_cache = {}

    def _stats(self, pair_idx: int, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if pair_idx not in self._stats_cache:
            self._stats_cache[pair_idx] = normalize_stats(compute_pair_stats(img1, img2))
        return self._stats_cache[pair_idx]

    def _run_and_eval(self):
        """Runs pipeline at self.n_kp and computes surrogate accuracy & utility."""
        r = pu.run_pipeline(
            self.img1, self.img2, self.preset, self.n_kp,
            matcher=MATCHER, ratio=RATIO,
            key1=self.key1, key2=self.key2
        )
        surrogate_acc, inlier_ratio = _surrogate_acc(r["pts1"], r["pts2"])

        self.effective_kp = r["n_kp"]
        self.pool_size = min(r["pool_n1"], r["pool_n2"])
        self.last_acc = surrogate_acc
        self.last_inlier_ratio = inlier_ratio
        self.last_n_matches = min(r["n_matches"], MATCH_NORM) / MATCH_NORM
        self.last_r = r

        cost = self.effective_kp / pu.N_MAX
        self.utility = self.last_acc - self.lam * cost
        return r

    def _update_best(self):
        """Maintains the best visited state across the episode."""
        if self.utility > self.best_utility:
            self.best_utility = self.utility
            self.best_n_kp = self.effective_kp
            self.best_pts1 = self.last_r["pts1"].copy()
            self.best_pts2 = self.last_r["pts2"].copy()

    def _obs(self) -> np.ndarray:
        onehot = np.zeros(N_PRESETS, dtype=np.float32)
        onehot[self.preset_idx] = 1.0
        extra = np.array([
            self.n_kp / pu.N_MAX,
            self.effective_kp / pu.N_MAX,
            self.pool_size / pu.N_MAX,
            self.last_acc,
            self.last_inlier_ratio,
            self.last_n_matches,
            self.steps / MAX_STEPS,
            self.delta_acc,
            self.delta_utility,
            self.is_plateau,
            self.gap_from_best,
        ], dtype=np.float32)
        obs = np.concatenate([self._pair_stats, onehot, extra]).astype(np.float32)
        assert obs.shape == (OBS_DIM,), f"Observation shape mismatch: {obs.shape}"
        return obs

    def _init_episode_state(self):
        self.prev_utility = self.utility
        self.prev_acc = self.last_acc
        self.delta_acc = 0.0
        self.delta_utility = 0.0
        self.is_plateau = float(self.effective_kp >= self.pool_size)
        self.best_utility = self.utility
        self.best_n_kp = self.effective_kp
        self.best_pts1 = self.last_r["pts1"].copy()
        self.best_pts2 = self.last_r["pts2"].copy()
        self.gap_from_best = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)

        self.pair_idx = self.rng.choice(self.pair_indices)
        item = self.dataset.load_images(self.pair_idx, grayscale=True)
        self.img1, self.img2 = item["img1"], item["img2"]
        self.K1, self.K2 = item["K1"], item["K2"]
        self.R_gt, self.T_gt = item["R_gt"], item["T_gt"]
        self.key1 = (f"md_{item['img1_rel']}", 1)
        self.key2 = (f"md_{item['img2_rel']}", 2)

        self.preset = self.rng.choice(pu.PRESET_NAMES)
        self.preset_idx = pu.PRESET_NAMES.index(self.preset)
        self.n_kp = pu.N_START
        self.steps = 0
        self._pair_stats = self._stats(self.pair_idx, self.img1, self.img2)

        self._run_and_eval()
        self._init_episode_state()
        return self._obs(), {}

    def reset_to(self, pair_idx: int, preset: str):
        """Deterministic reset to specific pair and preset for validation/testing."""
        self.pair_idx = pair_idx
        item = self.dataset.load_images(self.pair_idx, grayscale=True)
        self.img1, self.img2 = item["img1"], item["img2"]
        self.K1, self.K2 = item["K1"], item["K2"]
        self.R_gt, self.T_gt = item["R_gt"], item["T_gt"]
        self.key1 = (f"md_{item['img1_rel']}", 1)
        self.key2 = (f"md_{item['img2_rel']}", 2)

        self.preset = preset
        self.preset_idx = pu.PRESET_NAMES.index(preset)
        self.n_kp = pu.N_START
        self.steps = 0
        self._pair_stats = self._stats(self.pair_idx, self.img1, self.img2)

        self._run_and_eval()
        self._init_episode_state()
        return self._obs(), {}

    def _best_info(self):
        return {
            "best_n_kp": self.best_n_kp,
            "best_utility": round(self.best_utility, 4),
            "final_n_kp": self.n_kp,
            "steps_taken": self.steps,
        }

    def step(self, action: int):
        self.steps += 1
        delta = STEP_SIZES[action]

        if delta is None:  # STOP
            info = dict(n_kp=self.n_kp, preset=self.preset)
            info.update(self._best_info())
            return self._obs(), 0.0, True, False, info

        self.n_kp = int(np.clip(self.n_kp + delta, pu.N_MIN, pu.N_MAX))
        self._run_and_eval()

        truncated = self.steps >= MAX_STEPS
        terminated = False

        self.delta_acc = self.last_acc - self.prev_acc
        self.delta_utility = self.utility - self.prev_utility
        self.is_plateau = float(self.effective_kp >= self.pool_size)

        reward = self.delta_utility - STEP_PENALTY
        self.prev_utility = self.utility
        self.prev_acc = self.last_acc

        self._update_best()
        self.gap_from_best = self.utility - self.best_utility

        info = dict(n_kp=self.n_kp, preset=self.preset)
        info.update(self._best_info())
        return self._obs(), reward, terminated, truncated, info
