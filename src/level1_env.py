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

    def __init__(self, split="train", l2_model_path=DEFAULT_L2_MODEL_PATH, seed=None):
        super().__init__()
        self.split = split
        self.pairs = list_pairs(load_split()[split])
        self.rng = random.Random(seed)

        self.n_presets = len(pu.PRESET_NAMES)
        self.action_space = spaces.Discrete(self.n_presets)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(N_STATS,), dtype=np.float32)

        # Level 2 evaluator (frozen)
        self.l2_env = Level2Env(split=split, seed=seed)
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
            "best_utility": float(info["best_utility"]),
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

        reward = res["best_utility"]
        terminated = True
        truncated = False

        info = {
            "seq": self.seq,
            "k": self.k,
            "preset": preset,
            "action": action,
            "reward": reward,
            "best_utility": res["best_utility"],
            "best_n_kp": res["best_n_kp"],
            "best_h_ok": res["best_h_ok"],
            "best_corner_err": res["best_corner_err"],
        }
        return self.obs.copy(), reward, terminated, truncated, info

