import os
from pathlib import Path
import numpy as np
import gymnasium as gym
import pipeline_utils as pu
import megadepth as md
from level2_env import Level2Env, MATCHER, RATIO, MATCH_NORM
from image_stats import compute_pair_stats, normalize_stats

ACC_SCALE_DEG = 5.0      # accuracy = exp(-pose_err / 5 deg), same shape as exp(-corner_err / 5 px)
STATS_DIR = Path(__file__).resolve().parents[1] / "data" / "megadepth" / "stats_cache"


def pair_stats(fname, i, j, img1=None, img2=None):
    """Normalized 9-value image statistics for a pair, cached on disk."""
    k1, k2 = md.pair_keys(fname, i, j)
    path = STATS_DIR / f"{k1[0]}__{k1[1]}__{k2[1]}.npy"
    if path.exists():
        return np.load(path)
    if img1 is None:
        img1, img2, _, _, _ = md.load_pair(fname, i, j)
    s = normalize_stats(compute_pair_stats(img1, img2))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, s)
    os.replace(tmp, path)
    return s


class MegaDepthEnv(Level2Env):
    """Same observation (34 values), actions and bookkeeping as Level2Env.
    Only the accuracy changes: pose error (degrees) from ground-truth pose.
    Fields named corner_err / best_corner_err in info now hold the POSE ERROR."""

    def __init__(self, lam=0.1):
        super().__init__(split="val", lam=lam)     # HPatches pair list is unused

    def _run_and_eval(self):
        r = pu.run_pipeline(self.img1, self.img2, self.preset, self.n_kp,
                            matcher=MATCHER, ratio=RATIO, key1=self.key1, key2=self.key2)
        err, ratio = np.inf, 0.0
        if r["n_matches"] >= 8:
            est = md.estimate_pose_ex(r["pts1"].astype(np.float64), r["pts2"].astype(np.float64),
                                      self.K0, self.K1)
            if est is not None:
                err = float(md.pose_error(self.T, est[0], est[1]))
                ratio = est[2]
        acc = float(np.exp(-err / ACC_SCALE_DEG)) if np.isfinite(err) else 0.0
        m = dict(pose_err=err, corner_err=err, precision=ratio, ratio=ratio,
                 h_ok=float(err < 5.0), n_matches=r["n_matches"], reward_acc=acc)

        self.effective_kp = r["n_kp"]
        self.pool_size = min(r["pool_n1"], r["pool_n2"])
        self.last_acc = acc
        self.last_inlier_ratio = ratio
        self.last_n_matches = min(r["n_matches"], MATCH_NORM) / MATCH_NORM
        self.last_m = m
        self.utility = acc - self.lam * (self.effective_kp / pu.N_MAX)
        return m

    def reset_to(self, pair, preset):
        fname, i, j = pair
        self.img1, self.img2, self.K0, self.K1, self.T = md.load_pair(fname, i, j)
        self.key1, self.key2 = md.pair_keys(fname, i, j)
        self.preset = preset
        self.preset_idx = pu.PRESET_NAMES.index(preset)
        self.n_kp = pu.N_START
        self.steps = 0
        self._pair_stats = pair_stats(fname, i, j, self.img1, self.img2)
        self._run_and_eval()
        self._init_episode_state()
        return self._obs(), {}

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        pairs = md.list_pairs()
        pair = pairs[int(self.np_random.integers(len(pairs)))]
        preset = pu.PRESET_NAMES[int(self.np_random.integers(len(pu.PRESET_NAMES)))]
        return self.reset_to(pair, preset)