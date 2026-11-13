import os
import random
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from hpatches import load_split, load_pair, list_pairs
import pipeline_utils as pu
from image_stats import compute_pair_stats, normalize_stats, N_STATS
from level2_env import Level2Env

DEFAULT_L2_MODEL_PATH = "checkpoints/level2_14preset_best/best_model.zip"

class Level1Env(gym.Env):
    """
    Level 1 Pipeline Selection Environment (Contextual Bandit).

    Observation:
        9-dimensional visual pair statistics (entropy, blur, corner density, gradients, etc.)
    Action:
        Discrete(N_PRESETS): Selects which feature matching preset to use.
    Reward:
        The final utility achieved by the frozen downstream Level 2 agent:
        reward = best_utility = reward_acc - lambda * (n_kp / 4000)
    Episode:
        Single-step (Contextual Bandit): 1 action -> evaluation -> terminated = True.
    """
    metadata = {"render_modes": []}

    def __init__(self, split="train", l2_model_path=DEFAULT_L2_MODEL_PATH, seed=None, eval_mode=False, obs_mode="gt", split_file=None):
        super().__init__()
        self.split = split
        self.split_file = split_file
        self.pairs = list_pairs(load_split(split_file)[split])
        self.rng = random.Random(seed)
        self.eval_mode = eval_mode
        self.obs_mode = obs_mode

        self.n_presets = len(pu.PRESET_NAMES)
        self.action_space = spaces.Discrete(self.n_presets)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(N_STATS,), dtype=np.float32)

        # Level 2 evaluator (frozen)
        self.l2_env = Level2Env(split=split, seed=seed, eval_mode=eval_mode, obs_mode=obs_mode, split_file=split_file)
        if not os.path.exists(l2_model_path):
            # Fallback to deltaobs if 14preset isn't found
            alt_path = "checkpoints/level2_deltaobs_best/best_model.zip"
            if os.path.exists(alt_path):
                l2_model_path = alt_path
            else:
                raise FileNotFoundError(f"Cannot find Level 2 checkpoint at {l2_model_path}")
        
        self.l2_model = PPO.load(l2_model_path)
        
        # Caches to avoid re-evaluating identical (pair, preset) combinations
        self._stats_cache = {}
        self._eval_cache = {}

    def _stats(self, seq, k, img1, img2):
        key = (seq, k)
        if key not in self._stats_cache:
            self._stats_cache[key] = normalize_stats(compute_pair_stats(img1, img2))
        return self._stats_cache[key]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)

        self.seq, self.k = self.rng.choice(self.pairs)
        self.img1, self.img2, self.H = load_pair(self.seq, self.k)
        self.obs = self._stats(self.seq, self.k, self.img1, self.img2)
        return self.obs.copy(), {"seq": self.seq, "k": self.k}

    def reset_to(self, seq, k):
        """Deterministic reset to a specific image pair."""
        self.seq, self.k = seq, k
        self.img1, self.img2, self.H = load_pair(seq, k)
        self.obs = self._stats(seq, k, self.img1, self.img2)
        return self.obs.copy(), {"seq": self.seq, "k": self.k}

    def _run_level2(self, preset):
        cache_key = (self.seq, self.k, preset)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        obs, info = self.l2_env.reset_to(self.seq, self.k, preset)
        while True:
            action, _ = self.l2_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.l2_env.step(int(action))
            if terminated or truncated:
                break

        res = {
            "final_gt_utility": float(info["final_gt_utility"]),
            "best_gt_utility": float(info["best_gt_utility"]),
            "final_n_kp": int(info["final_n_kp"]),
            "final_h_ok": float(info["final_h_ok"]),
            "final_corner_err": float(info["final_corner_err"]),
            "final_precision": float(info.get("final_precision", 0.0)),
            "best_n_kp": int(info["best_n_kp"]),
            "best_h_ok": float(info["best_h_ok"]),
            "best_corner_err": float(info["best_corner_err"]),
            "best_precision": float(info.get("best_precision", 0.0)),
        }
        self._eval_cache[cache_key] = res
        return res

    def step(self, action):
        preset = pu.PRESET_NAMES[action]
        res = self._run_level2(preset)

        # Intended design: Level 1 reward is the GT-based utility of the strict-STOP final state
        # reported by the frozen Level 2 rollout (run with obs_mode="proxy").
        reward = res["final_gt_utility"]
        terminated = True
        truncated = False

        info = {
            "seq": self.seq,
            "k": self.k,
            "preset": preset,
            "action": action,
            "reward": reward,
            "final_gt_utility": res["final_gt_utility"],
            "best_gt_utility": res["best_gt_utility"],
            "final_n_kp": res["final_n_kp"],
            "best_n_kp": res["best_n_kp"],
            "final_h_ok": res["final_h_ok"],
            "best_h_ok": res["best_h_ok"],
            "final_corner_err": res["final_corner_err"],
            "best_corner_err": res["best_corner_err"],
        }
        return self.obs.copy(), reward, terminated, truncated, info

