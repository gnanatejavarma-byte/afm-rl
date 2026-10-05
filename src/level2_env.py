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
STEP_SIZES = [500, 1000, -500, None]   # None = STOP
MAX_STEPS = 15
DEFAULT_LAMBDA = 0.1
STEP_PENALTY = 0.01
MATCH_NORM = 1000.0

N_PRESETS = len(pu.PRESET_NAMES)
OBS_DIM = N_STATS + N_PRESETS + 11
# +11 = requested_n_kp, effective_kp, pool_size, last_acc,
#       last_inlier_ratio, last_n_matches, step_fraction,
#       delta_acc, delta_utility, is_plateau, gap_from_best


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

        self.effective_kp = r["n_kp"]
        self.pool_size = min(r["pool_n1"], r["pool_n2"])
        self.last_acc = m["reward_acc"]
        self.last_inlier_ratio = m["ransac_ratio"]
        self.last_n_matches = min(m["n_matches"], MATCH_NORM) / MATCH_NORM
        self.last_m = m

        cost = self.effective_kp / pu.N_MAX
        self.utility = self.last_acc - self.lam * cost
        return m

    def _update_best(self):
        """Scoreboard: remember the best (N, utility, metrics) seen THIS
        episode, regardless of where the agent ends up. This is plain
        bookkeeping, not something the agent has to learn -- so the final
        reported result can never be worse than the best point visited."""
        if self.utility > self.best_utility:
            self.best_utility = self.utility
            self.best_n_kp = self.effective_kp
            self.best_m = self.last_m

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
            self.delta_acc,       # did accuracy improve on the LAST step?
            self.delta_utility,   # did overall utility improve on the LAST step?
            self.is_plateau,      # 1.0 if the detector's pool is already maxed out
            self.gap_from_best,   # how far BELOW this episode's own best are we right now? (<=0)
        ], dtype=np.float32)
        return np.concatenate([self._pair_stats, onehot, extra]).astype(np.float32)

    def _init_episode_state(self):
        """Shared setup after the first _run_and_eval() of an episode."""
        self.prev_utility = self.utility
        self.prev_acc = self.last_acc
        self.delta_acc = 0.0        # no previous step exists yet
        self.delta_utility = 0.0
        self.is_plateau = float(self.effective_kp >= self.pool_size)
        self.best_utility = self.utility
        self.best_n_kp = self.effective_kp
        self.best_m = self.last_m
        self.gap_from_best = 0.0    # current point IS the best so far, by definition

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

        self._run_and_eval()
        self._init_episode_state()
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
        self._init_episode_state()
        return self._obs(), {}

    def _best_info(self):
        """What we actually report as 'the answer' -- the best point visited
        this episode, never just wherever the agent happened to stop."""
        bm = self.best_m
        return dict(best_n_kp=self.best_n_kp, best_utility=round(self.best_utility, 4),
                    best_precision=bm["precision"], best_h_ok=bm["h_ok"],
                    best_corner_err=bm["corner_err"])

    def step(self, action):
        self.steps += 1
        delta = STEP_SIZES[action]

        if delta is None:   # STOP: no recompute, no extra tax
            info = dict(n_kp=self.n_kp, preset=self.preset,
                        precision=self.last_m["precision"],
                        h_ok=self.last_m["h_ok"],
                        corner_err=self.last_m["corner_err"])
            info.update(self._best_info())
            return self._obs(), 0.0, True, False, info

        self.n_kp = int(np.clip(self.n_kp + delta, pu.N_MIN, pu.N_MAX))
        m = self._run_and_eval()

        truncated = self.steps >= MAX_STEPS
        terminated = False

        # Give the policy a memory of what its LAST action did, and how far
        # below its own best result it currently stands.
        self.delta_acc = self.last_acc - self.prev_acc
        self.delta_utility = self.utility - self.prev_utility
        self.is_plateau = float(self.effective_kp >= self.pool_size)

        # Reward stays on the RAW step-to-step delta -- NOT best-to-best.
        # Rewarding only new bests would make most exploratory steps score
        # exactly 0, starving PPO of the dense gradient it needs.
        reward = self.delta_utility - STEP_PENALTY
        self.prev_utility = self.utility
        self.prev_acc = self.last_acc

        self._update_best()   # scoreboard update happens AFTER reward is computed,
                               # using this step's true utility
        self.gap_from_best = self.utility - self.best_utility   # <= 0 always; 0 if this IS the new best

        info = dict(n_kp=self.n_kp, preset=self.preset,
                    precision=m["precision"], h_ok=m["h_ok"], corner_err=m["corner_err"])
        info.update(self._best_info())
        return self._obs(), reward, terminated, truncated, info