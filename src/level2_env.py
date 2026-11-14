import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hpatches import load_split, load_pair, list_pairs
import pipeline_utils as pu
from metrics import evaluate_matches, compute_label_free_proxy, get_ransac_seed
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


def build_level2_obs(pair_stats, preset_idx, n_presets, n_kp, effective_kp,
                     pool_size, last_acc, last_inlier_ratio, last_n_matches,
                     steps, delta_acc, delta_utility, is_plateau, gap_from_best,
                     max_steps=MAX_STEPS):
    onehot = np.zeros(n_presets, dtype=np.float32)
    onehot[preset_idx] = 1.0
    extra = np.array([
        n_kp / pu.N_MAX,
        effective_kp / pu.N_MAX,
        pool_size / pu.N_MAX,
        last_acc,
        last_inlier_ratio,
        last_n_matches,
        steps / max_steps,
        delta_acc,
        delta_utility,
        is_plateau,
        gap_from_best,
    ], dtype=np.float32)
    return np.concatenate([pair_stats, onehot, extra]).astype(np.float32)


class Level2Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split="train", lam=DEFAULT_LAMBDA, seed=None, eval_mode=False, obs_mode="gt", split_file=None):
        super().__init__()
        self.split = split
        self.split_file = split_file
        self.pairs = list_pairs(load_split(split_file)[split])
        self.lam = lam
        self.rng = random.Random(seed)
        self.eval_mode = eval_mode
        if obs_mode not in ("gt", "proxy"):
            raise ValueError(f"obs_mode must be 'gt' or 'proxy', got {obs_mode}")
        self.obs_mode = obs_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
        self._stats_cache = {}
        self._H = None
        self._in_eval_rollout = False

    @property
    def H(self):
        if getattr(self, "_in_eval_rollout", False):
            raise RuntimeError("CRITICAL LEAK DETECTED: self.H was accessed during an eval_mode episode rollout!")
        return self._H

    @H.setter
    def H(self, value):
        self._H = value

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
        self.effective_kp = r["n_kp"]
        self.pool_size = min(r["pool_n1"], r["pool_n2"])
        self.last_r = r
        cost = self.effective_kp / pu.N_MAX

        if self.obs_mode == "proxy":
            # Observation and Scoreboard are 100% computed from unsupervised proxy in BOTH train & eval
            r_seed = get_ransac_seed(self.seq, self.k, f"{self.preset}_step_{self.steps}") if self.eval_mode else None
            m_proxy = compute_label_free_proxy(r["pts1"], r["pts2"], self.img1.shape, ransac_seed=r_seed)
            self.last_acc = m_proxy["proxy"]
            self.last_inlier_ratio = m_proxy["inlier_ratio"]
            self.last_n_matches = min(m_proxy["n_matches"], MATCH_NORM) / MATCH_NORM
            self.last_m = m_proxy
            self.proxy_utility = self.last_acc - self.lam * cost
            self.utility = self.proxy_utility

            if not self.eval_mode:
                # In training mode with obs_mode="proxy": GT is used ONLY to compute the reward utility
                m_gt = evaluate_matches(r["pts1"], r["pts2"], self.H, self.img1.shape)
                self.gt_utility = m_gt["reward_acc"] - self.lam * cost
                m = m_gt
            else:
                self.gt_utility = 0.0
                m = m_proxy

        else: # self.obs_mode == "gt" (exact old behavior)
            if self.eval_mode:
                # 100% LABEL-FREE: Zero ground truth is accessed during evaluation rollout
                r_seed = get_ransac_seed(self.seq, self.k, f"{self.preset}_step_{self.steps}")
                m = compute_label_free_proxy(r["pts1"], r["pts2"], self.img1.shape, ransac_seed=r_seed)
                self.last_acc = m["proxy"]
                self.last_inlier_ratio = m["inlier_ratio"]
                self.last_n_matches = min(m["n_matches"], MATCH_NORM) / MATCH_NORM
                self.last_m = m
            else:
                # TRAINING ONLY: Supervised RL with geometric reward
                m = evaluate_matches(r["pts1"], r["pts2"], self.H, self.img1.shape)
                self.last_acc = m["reward_acc"]
                self.last_inlier_ratio = m["ransac_ratio"]
                self.last_n_matches = min(m["n_matches"], MATCH_NORM) / MATCH_NORM
                self.last_m = m

            self.utility = self.last_acc - self.lam * cost
            self.gt_utility = self.utility

        return m

    def _update_best(self):
        """Scoreboard: remember the best (N, utility, metrics) seen THIS
        episode, regardless of where the agent ends up."""
        if self.utility > self.best_utility:
            self.best_utility = self.utility
            self.best_n_kp = self.effective_kp
            self.best_m = self.last_m
            self.best_r = self.last_r

    def _obs(self):
        return build_level2_obs(
            self._pair_stats,
            self.preset_idx,
            N_PRESETS,
            self.n_kp,
            self.effective_kp,
            self.pool_size,
            self.last_acc,
            self.last_inlier_ratio,
            self.last_n_matches,
            self.steps,
            self.delta_acc,
            self.delta_utility,
            self.is_plateau,
            self.gap_from_best,
            MAX_STEPS,
        )

    def _init_episode_state(self):
        """Shared setup after the first _run_and_eval() of an episode."""
        self.prev_utility = self.utility
        self.prev_acc = self.last_acc
        self.prev_gt_utility = getattr(self, "gt_utility", self.utility)
        self.delta_acc = 0.0        # no previous step exists yet
        self.delta_utility = 0.0
        self.is_plateau = float(self.effective_kp >= self.pool_size)
        self.best_utility = self.utility
        self.best_n_kp = self.effective_kp
        self.best_m = self.last_m
        self.best_r = self.last_r
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

        if self.eval_mode:
            self._in_eval_rollout = True

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

        if self.eval_mode:
            self._in_eval_rollout = True

        self._run_and_eval()
        self._init_episode_state()
        return self._obs(), {}

    def _best_info(self):
        """What we report as 'the answer'. In eval_mode (or obs_mode='proxy'), evaluate_matches is run ONCE
        post-hoc on the selected points to record benchmark statistics (AUC, corner_err)."""
        if self.eval_mode or self.obs_mode == "proxy":
            s_best = get_ransac_seed(self.seq, self.k, f"{self.preset}_score_best") if self.eval_mode else None
            s_final = get_ransac_seed(self.seq, self.k, f"{self.preset}_score_final") if self.eval_mode else None
            bm = evaluate_matches(self.best_r["pts1"], self.best_r["pts2"], self.H, self.img1.shape, ransac_seed=s_best)
            fm = evaluate_matches(self.last_r["pts1"], self.last_r["pts2"], self.H, self.img1.shape, ransac_seed=s_final)
            best_gt_u = float(bm["reward_acc"] - self.lam * (self.best_n_kp / pu.N_MAX))
            final_gt_u = float(fm["reward_acc"] - self.lam * (self.effective_kp / pu.N_MAX))
            return dict(
                best_n_kp=self.best_n_kp,
                best_utility=round(best_gt_u, 4),
                best_gt_utility=best_gt_u,
                best_proxy_utility=round(self.best_utility, 4),
                best_precision=bm["precision"],
                best_h_ok=bm["h_ok"],
                best_corner_err=bm["corner_err"],
                final_n_kp=self.effective_kp,
                final_utility=round(final_gt_u, 4),
                final_gt_utility=final_gt_u,
                final_precision=fm["precision"],
                final_h_ok=fm["h_ok"],
                final_corner_err=fm["corner_err"],
            )
        else:
            bm = self.best_m
            fm = self.last_m
            return dict(
                best_n_kp=self.best_n_kp,
                best_utility=round(self.best_utility, 4),
                best_gt_utility=float(self.best_utility),
                best_precision=bm.get("precision", 0.0),
                best_h_ok=bm.get("h_ok", 0.0),
                best_corner_err=bm.get("corner_err", float("inf")),
                final_n_kp=self.effective_kp,
                final_utility=round(self.utility, 4),
                final_gt_utility=float(self.utility),
                final_precision=fm.get("precision", 0.0),
                final_h_ok=fm.get("h_ok", 0.0),
                final_corner_err=fm.get("corner_err", float("inf")),
            )

    def step(self, action):
        self.steps += 1
        delta = STEP_SIZES[action]

        if delta is None:   # STOP: no recompute, no extra tax
            self._in_eval_rollout = False  # episode finished
            if self.eval_mode:
                info = dict(n_kp=self.effective_kp, preset=self.preset)
            else:
                info = dict(n_kp=self.n_kp, preset=self.preset,
                            precision=self.last_m.get("precision", 0.0),
                            h_ok=self.last_m.get("h_ok", 0.0),
                            corner_err=self.last_m.get("corner_err", float("inf")))
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
        if self.eval_mode:
            reward = self.delta_utility - STEP_PENALTY
        else:
            delta_gt_u = self.gt_utility - self.prev_gt_utility
            reward = delta_gt_u - STEP_PENALTY

        self.prev_utility = self.utility
        self.prev_acc = self.last_acc
        self.prev_gt_utility = getattr(self, "gt_utility", self.utility)

        self._update_best()   # scoreboard update happens AFTER reward is computed,
                               # using this step's true utility
        self.gap_from_best = self.utility - self.best_utility   # <= 0 always; 0 if this IS the new best

        if self.eval_mode:
            info = dict(n_kp=self.effective_kp, preset=self.preset,
                        inlier_ratio=self.last_inlier_ratio, proxy=self.last_acc)
        else:
            info = dict(n_kp=self.n_kp, preset=self.preset,
                        precision=m.get("precision", 0.0), h_ok=m.get("h_ok", 0.0),
                        corner_err=m.get("corner_err", float("inf")))

        if terminated or truncated:
            self._in_eval_rollout = False  # episode finished
            info.update(self._best_info())
        return self._obs(), reward, terminated, truncated, info