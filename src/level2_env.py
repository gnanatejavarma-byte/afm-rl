import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hpatches import load_split, load_pair, list_pairs
import pipeline_utils as pu
from metrics import evaluate_matches
from image_stats import compute_pair_stats, normalize_stats, N_STATS

MATCHER = "BF"
RATIO = 0.75
STEP_SIZES = [500, 1000, -500, None]   # None = STOP; +-100 dropped: proven too weak
                                        # a signal to ever beat any nonzero step cost
MAX_STEPS = 15
DEFAULT_LAMBDA = 0.1
STEP_PENALTY = 0.01     # lowered: PBRS reward is a genuine utility delta now,
                        # so the tax only needs to rule out truly useless moves
MATCH_NORM = 1000.0

N_PRESETS = len(pu.PRESET_NAMES)
OBS_DIM = N_STATS + N_PRESETS + 7
# +7 = requested_n_kp, effective_kp, pool_size, last_acc,
#      last_inlier_ratio, last_n_matches, step_fraction


class Level2Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split="train", lam=DEFAULT_LAMBDA, seed=None):
        super().__init__()
        self.pairs = list_pairs(load_split()[split])
        self.lam = lam
        self.rng = random.Random(seed)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
        self._stats_cache = {}

    def _stats(self, seq, k, img1, img2):
        key = (seq, k)
        if key not in self._stats_cache:
            self._stats_cache[key] = normalize_stats(compute_pair_stats(img1, img2))
        return self._stats_cache[key]

    def _run_and_eval(self):
        """Run the pipeline at self.n_kp; store everything needed for both the
        reward (utility) and the observation."""
        r = pu.run_pipeline(self.img1, self.img2, self.preset, self.n_kp,
                             matcher=MATCHER, ratio=RATIO,
                             key1=self.key1, key2=self.key2)
        m = evaluate_matches(r["pts1"], r["pts2"], self.H, self.img1.shape)

        self.effective_kp = r["n_kp"]                     # min(requested, pool), from pipeline_utils
        self.pool_size = min(r["pool_n1"], r["pool_n2"])  # detector's real ceiling for this pair+preset
        self.last_acc = m["reward_acc"]
        self.last_inlier_ratio = m["ransac_ratio"]
        self.last_n_matches = min(m["n_matches"], MATCH_NORM) / MATCH_NORM
        self.last_m = m   # cached so STOP can report diagnostics without recomputing

        cost = self.effective_kp / pu.N_MAX   # charge cost on REAL keypoints used, not phantom requested N
        self.utility = self.last_acc - self.lam * cost
        return m

    def _obs(self):
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
        ], dtype=np.float32)
        return np.concatenate([self._pair_stats, onehot, extra]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        seq, k = self.rng.choice(self.pairs)
        self.seq, self.k = seq, k
        self.img1, self.img2, self.H = load_pair(seq, k)
        self.key1, self.key2 = (seq, 1), (seq, k)
        self.preset = self.rng.choice(pu.PRESET_NAMES)
        self.preset_idx = pu.PRESET_NAMES.index(self.preset)
        self.n_kp = pu.N_START
        self.steps = 0
        self._pair_stats = self._stats(seq, k, self.img1, self.img2)

        self._run_and_eval()          # honest baseline: what does N_START actually achieve?
        self.prev_utility = self.utility
        return self._obs(), {}

    def reset_to(self, seq, k, preset):
        """Deterministic reset to a SPECIFIC pair+preset, for inspection/evaluation."""
        self.seq, self.k = seq, k
        self.img1, self.img2, self.H = load_pair(seq, k)
        self.key1, self.key2 = (seq, 1), (seq, k)
        self.preset = preset
        self.preset_idx = pu.PRESET_NAMES.index(preset)
        self.n_kp = pu.N_START
        self.steps = 0
        self._pair_stats = self._stats(seq, k, self.img1, self.img2)
        self._run_and_eval()
        self.prev_utility = self.utility
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        delta = STEP_SIZES[action]

        if delta is None:   # STOP: no recompute, no extra tax -- agent already
                             # "banked" whatever utility it reached
            info = dict(n_kp=self.n_kp, preset=self.preset,
                        precision=self.last_m["precision"],
                        h_ok=self.last_m["h_ok"],
                        corner_err=self.last_m["corner_err"])
            return self._obs(), 0.0, True, False, info

        self.n_kp = int(np.clip(self.n_kp + delta, pu.N_MIN, pu.N_MAX))
        m = self._run_and_eval()

        truncated = self.steps >= MAX_STEPS
        terminated = False

        # Potential-based shaping: reward = genuine utility gained this step,
        # minus a small exploration tax. Telescopes to
        # (final utility - starting utility) - (T-1)*STEP_PENALTY over the episode.
                # flat per step: RANSAC/matching costs roughly the same regardless of
        # how big the keypoint jump was, so the tax shouldn't scale with it
        reward = (self.utility - self.prev_utility) - STEP_PENALTY
        self.prev_utility = self.utility   # <-- critical: must update every step,
                                            # or later steps get compared to the
                                            # wrong baseline (the original reset value)
        info = dict(n_kp=self.n_kp, preset=self.preset,
                    precision=m["precision"], h_ok=m["h_ok"], corner_err=m["corner_err"])
        return self._obs(), reward, terminated, truncated, info