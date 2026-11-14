import os
import sys
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

import pipeline_utils as pu
from level2_env import Level2Env, build_level2_obs, OBS_DIM, STEP_SIZES, MAX_STEPS, MATCH_NORM
from megadepth_eval_env import MegaDepthEvalEnv

# Import old megadepth_eval_env for comparison
old_spec = importlib.util.spec_from_file_location("old_megadepth_eval_env", "results/phase5b/old_megadepth_eval_env.py")
old_mod = importlib.util.module_from_spec(old_spec)
old_spec.loader.exec_module(old_mod)
OldMegaDepthEvalEnv = old_mod.MegaDepthEvalEnv

def run_parity_test():
    print("=" * 70)
    print("RUNNING PARITY TEST: Level2Env vs MegaDepthEvalEnv (New vs Old)")
    print("=" * 70)

    # 1. Generate 10 test states (5 HPatches-like, 5 MegaDepth-like)
    np.random.seed(42)
    presets_pool = ["SIFT", "ALIKED", "ORB", "XFeat", "AKAZE", "SIFT", "ALIKED", "FAST+BRIEF", "GFTT+SIFT", "ORB"]
    
    test_states = []
    for idx in range(10):
        pair_type = "HPatches" if idx < 5 else "MegaDepth"
        pair_stats = np.random.uniform(-2.0, 2.0, 9).astype(np.float32)
        preset = presets_pool[idx]
        preset_idx = pu.PRESET_NAMES.index(preset)
        n_kp = int(np.random.choice([500, 1000, 1500, 2000, 3000]))
        effective_kp = min(n_kp, int(np.random.choice([400, 500, 1000, 2000, 3500])))
        pool_size = 4000
        last_acc = float(np.random.uniform(0.1, 0.9))
        last_inlier_ratio = float(np.random.uniform(0.05, 0.8))
        raw_matches = int(np.random.uniform(20, 800))
        last_n_matches = float(min(raw_matches, MATCH_NORM) / MATCH_NORM)
        steps = int(np.random.choice([0, 1, 2, 3, 4]))
        delta_acc = float(np.random.uniform(-0.1, 0.1))
        delta_utility = float(np.random.uniform(-0.05, 0.05))
        is_plateau = float(effective_kp >= pool_size)
        gap_from_best = float(-np.abs(np.random.uniform(0.0, 0.2)))
        
        test_states.append({
            "idx": idx,
            "pair_type": pair_type,
            "pair_stats": pair_stats,
            "preset": preset,
            "preset_idx": preset_idx,
            "n_kp": n_kp,
            "effective_kp": effective_kp,
            "pool_size": pool_size,
            "last_acc": last_acc,
            "last_inlier_ratio": last_inlier_ratio,
            "last_n_matches": last_n_matches,
            "steps": steps,
            "delta_acc": delta_acc,
            "delta_utility": delta_utility,
            "is_plateau": is_plateau,
            "gap_from_best": gap_from_best
        })

    # 2. Test Observations
    print("\n--- TEST 1: Observation Vector Dimension-by-Dimension Parity ---")
    
    old_obs_failures = 0
    new_obs_passes = 0
    sample_34dim_obs = None
    
    for s in test_states:
        # Ground truth observation from Level2Env's logic
        obs_l2 = build_level2_obs(
            s["pair_stats"], s["preset_idx"], len(pu.PRESET_NAMES),
            s["n_kp"], s["effective_kp"], s["pool_size"],
            s["last_acc"], s["last_inlier_ratio"], s["last_n_matches"],
            s["steps"], s["delta_acc"], s["delta_utility"],
            s["is_plateau"], s["gap_from_best"], MAX_STEPS
        )
        if sample_34dim_obs is None:
            sample_34dim_obs = (s, obs_l2)
            
        # New MegaDepth env observation logic
        env_new = MegaDepthEvalEnv.__new__(MegaDepthEvalEnv)
        env_new.pair_stats = s["pair_stats"]
        env_new.preset_idx = s["preset_idx"]
        env_new.n_kp = s["n_kp"]
        env_new.effective_kp = s["effective_kp"]
        env_new.pool_size = s["pool_size"]
        env_new.last_acc = s["last_acc"]
        env_new.last_proxy = s["last_acc"]
        env_new.last_inlier_ratio = s["last_inlier_ratio"]
        env_new.last_n_matches = s["last_n_matches"]
        env_new.steps = s["steps"]
        env_new.delta_acc = s["delta_acc"]
        env_new.delta_utility = s["delta_utility"]
        env_new.is_plateau = s["is_plateau"]
        env_new.gap_from_best = s["gap_from_best"]
        
        obs_new = env_new._get_obs()
        
        # Old MegaDepth env observation logic
        env_old = OldMegaDepthEvalEnv.__new__(OldMegaDepthEvalEnv)
        env_old.pair_stats = s["pair_stats"]
        env_old.preset_idx = s["preset_idx"]
        env_old.n_kp = s["n_kp"]
        env_old.effective_kp = s["effective_kp"]
        env_old.pool_size = s["pool_size"]
        env_old.last_proxy = s["last_acc"]
        env_old.prev_proxy = s["last_acc"] - s["delta_acc"]
        env_old.last_inlier_ratio = s["last_inlier_ratio"]
        env_old.prev_inlier_ratio = s["last_inlier_ratio"]
        env_old.last_n_matches = s["last_n_matches"]
        env_old.prev_n_matches = s["last_n_matches"]
        env_old.steps = s["steps"]
        
        obs_old = env_old._get_obs()
        
        # Check agreement
        diff_new = np.max(np.abs(obs_l2 - obs_new))
        diff_old = np.max(np.abs(obs_l2 - obs_old))
        
        if diff_new == 0.0:
            new_obs_passes += 1
        if diff_old > 1e-4:
            old_obs_failures += 1
            
    print(f"Old code observation match failures: {old_obs_failures}/10 (Expected: 10/10)")
    print(f"New code observation bit-identical matches: {new_obs_passes}/10 (Expected: 10/10)")
    assert old_obs_failures == 10, "Old code was expected to fail on all 10 states!"
    assert new_obs_passes == 10, "New code was expected to pass bit-identically on all 10 states!"

    # 3. Test Action Transitions and Semantics from N in {100, 500, 4000}
    print("\n--- TEST 2: Action Transitions (N_after, terminated) Parity ---")
    actions = [0, 1, 2, 3]
    test_Ns = [100, 500, 4000]
    
    # Expected Level2Env semantics
    # Action 0: +500
    # Action 1: +1000
    # Action 2: -500
    # Action 3: STOP
    
    old_action_mismatches = 0
    new_action_matches = 0
    total_action_tests = 0
    
    for n_start in test_Ns:
        for a in actions:
            total_action_tests += 1
            delta = STEP_SIZES[a]
            if delta is None:
                exp_n = n_start
                exp_term = True
            else:
                exp_n = int(np.clip(n_start + delta, pu.N_MIN, pu.N_MAX))
                exp_term = False
                
            # Test New Env Step logic (mocking _run_and_eval)
            env_new = MegaDepthEvalEnv.__new__(MegaDepthEvalEnv)
            env_new._in_eval_rollout = True
            env_new.steps = 0
            env_new.n_kp = n_start
            env_new.cum_kp = 0
            env_new.last_acc = 0.5
            env_new.prev_acc = 0.5
            env_new.utility = 0.5
            env_new.prev_utility = 0.5
            env_new.best_utility = 0.5
            env_new.last_proxy = 0.5
            env_new.effective_kp = n_start
            env_new.pool_size = 4000
            env_new._get_info = lambda: {}
            env_new._get_obs = lambda: np.zeros(34)
            env_new._run_and_eval = lambda: None
            
            _, _, term_new, trunc_new, _ = env_new.step(a)
            n_after_new = env_new.n_kp
            
            # Test Old Env Step logic
            env_old = OldMegaDepthEvalEnv.__new__(OldMegaDepthEvalEnv)
            env_old._in_eval_rollout = True
            env_old.steps = 0
            env_old.n_kp = n_start
            env_old.cum_kp = 0
            env_old.last_proxy = 0.5
            env_old.prev_proxy = 0.5
            env_old.last_inlier_ratio = 0.5
            env_old.prev_inlier_ratio = 0.5
            env_old.last_n_matches = 0.5
            env_old.prev_n_matches = 0.5
            env_old.proxy_utility = 0.5
            env_old.best_proxy_utility = 0.5
            env_old.effective_kp = n_start
            env_old.pool_size = 4000
            env_old._get_info = lambda: {}
            env_old._get_obs = lambda: np.zeros(34)
            env_old._run_and_eval = lambda: None
            
            _, _, term_old, trunc_old, _ = env_old.step(a)
            n_after_old = env_old.n_kp
            
            if (n_after_new == exp_n) and (term_new == exp_term):
                new_action_matches += 1
            else:
                print(f"NEW ACTION MISMATCH: Action {a} from N={n_start}: got ({n_after_new}, {term_new}) vs exp ({exp_n}, {exp_term})")
                
            if (n_after_old != exp_n) or (term_old != exp_term):
                old_action_mismatches += 1
                
    print(f"Old code action mismatches: {old_action_mismatches}/{total_action_tests} (Actions 0, 1, 3 have swapped semantics; Action 1 clamps at N=4000, Action 2 is -500 in both)")
    print(f"New code action exact matches: {new_action_matches}/{total_action_tests} (Expected: 12/12 matches)")
    assert old_action_mismatches == 8, f"Old code was expected to mismatch on 8 transitions, got {old_action_mismatches}!"
    assert new_action_matches == 12, "New code was expected to match all action transitions!"

    # 4. Print one full 34-dim observation vector
    print("\n--- SAMPLE 34-DIMENSION OBSERVATION VECTOR ---")
    sample_state, sample_vec = sample_34dim_obs
    print(f"Sample State: Pair Type={sample_state['pair_type']}, Preset={sample_state['preset']}, N={sample_state['n_kp']}, Steps={sample_state['steps']}")
    
    dim_names = [
        "stat_0_avg_texture", "stat_1_tex_change", "stat_2_avg_blur", "stat_3_blur_change",
        "stat_4_avg_corner_density", "stat_5_corner_change", "stat_6_avg_grad_mag",
        "stat_7_grad_spread", "stat_8_intensity_diff"
    ] + [f"onehot_{p}" for p in pu.PRESET_NAMES] + [
        "budget_norm", "effective_norm", "pool_norm", "last_acc",
        "last_inlier_ratio", "last_n_matches", "step_fraction",
        "delta_acc", "delta_utility", "is_plateau", "gap_from_best"
    ]
    
    df_obs = pd.DataFrame({
        "dim_idx": list(range(34)),
        "dim_name": dim_names,
        "value": sample_vec
    })
    print(df_obs.to_string(index=False))
    
    # Save parity test results table
    out_csv = Path("results/phase5b/parity_test_sample_vector.csv")
    df_obs.to_csv(out_csv, index=False)
    print(f"\nSaved sample vector to {out_csv}")
    print("\nPARITY TEST STATUS: 100% PASS (FAILS ON OLD CODE, PASSES ON NEW CODE)")

if __name__ == "__main__":
    run_parity_test()
