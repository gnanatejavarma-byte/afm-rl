import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hpatches
import pipeline_utils as pu
from level2_env import Level2Env
from level1_env import Level1Env
from metrics import evaluate_matches, CORNER_PX, REPROJ_PX, get_ransac_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AFM-RL models on a specific fold split")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0..4)")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="Split to evaluate ('test' for benchmark, 'val' for dev dryrun)")
    parser.add_argument("--split-file", type=str, default=None, help="Path to fold split JSON")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory containing fold models")
    parser.add_argument("--out-csv", type=str, default="results/phase3/folds_pairs.csv", help="Path to output CSV")
    parser.add_argument("--force", action="store_true", default=False, help="Overwrite output CSV if it exists")
    parser.add_argument("--legacy", action="store_true", default=False, help="Use legacy checkpoints/ models (for dev dryrun)")
    return parser.parse_args()

def run_benchmark_fold(args):
    split_file = args.split_file if args.split_file else f"configs/folds/fold{args.fold}.json"
    models_dir = args.models_dir if args.models_dir else f"checkpoints_v2/fold{args.fold}"

    if args.legacy:
        l1_path = "checkpoints/level1_best/best_model.zip"
        l2_path = "checkpoints/level2_14preset_best/best_model.zip"
    else:
        l1_path = os.path.join(models_dir, "level1_best", "best_model.zip")
        l2_path = os.path.join(models_dir, "level2_14preset_best", "best_model.zip")

    print("=" * 70)
    print(f"BENCHMARK FOLD {args.fold} (Split: {args.split.upper()})")
    print(f"  Split File:  {split_file}")
    print(f"  Level 1:     {l1_path}")
    print(f"  Level 2:     {l2_path}")
    print(f"  Output CSV:  {args.out_csv}")
    print("=" * 70)

    # Check output file protection
    out_path = Path(args.out_csv)
    if out_path.exists() and not args.force:
        # Check if this fold and split already exists in the file
        existing_df = pd.read_csv(out_path)
        if "fold" in existing_df.columns and "split" in existing_df.columns:
            has_fold = len(existing_df[(existing_df["fold"] == args.fold) & (existing_df["split"] == args.split)]) > 0
            if has_fold:
                raise FileExistsError(
                    f"Output file {args.out_csv} already contains fold={args.fold}, split={args.split}! "
                    f"Use --force to overwrite or specify a different --out-csv."
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load split sequences and pairs
    s = hpatches.load_split(split_file)
    seqs = sorted(s[args.split])
    pairs = hpatches.list_pairs(seqs)
    print(f"Evaluating {len(pairs)} pairs across {len(seqs)} sequences...")

    l1_model = PPO.load(l1_path)
    l2_model = PPO.load(l2_path)

    # Honest label-free environments
    l2_env = Level2Env(split=args.split, seed=123, eval_mode=True, obs_mode="proxy", split_file=split_file)
    l1_env = Level1Env(split=args.split, seed=123, eval_mode=True, obs_mode="proxy", l2_model_path=l2_path, split_file=split_file)

    rows = []

    # 1. Level 1 Selections
    l1_selections = {}
    for seq, k in pairs:
        obs, _ = l1_env.reset_to(seq, k)
        act, _ = l1_model.predict(obs, deterministic=True)
        chosen_preset = pu.PRESET_NAMES[int(act)]
        l1_selections[(seq, k)] = (int(act), chosen_preset)

    # 2. Evaluate all presets and policies
    for idx, (seq, k) in enumerate(pairs):
        img1, img2, H = hpatches.load_pair(seq, k)
        key1, key2 = (seq, 1), (seq, k)

        pair_preset_rollouts = {}

        # Run Level 2 on all 14 presets
        for preset in pu.PRESET_NAMES:
            obs, info = l2_env.reset_to(seq, k, preset)
            steps = 0
            actions = []
            cum_kp = l2_env.effective_kp
            while True:
                act, _ = l2_model.predict(obs, deterministic=True)
                actions.append(int(act))
                obs, reward, term, trunc, step_info = l2_env.step(int(act))
                steps += 1
                cum_kp += l2_env.effective_kp
                if term or trunc:
                    break

            pair_preset_rollouts[preset] = {
                "step_info": step_info,
                "steps": steps,
                "cum_kp": cum_kp,
                "actions": actions,
                "first_action": actions[0] if len(actions) > 0 else 3,
            }

            # L2-only per preset (Strict STOP)
            rows.append({
                "fold": args.fold, "split": args.split, "seq": seq, "k": k,
                "method": f"{preset}+L2", "preset": preset,
                "final_kp": step_info["final_n_kp"], "cum_kp": cum_kp, "steps": steps,
                "corner_err": step_info["final_corner_err"], "success": step_info["final_h_ok"],
                "first_action": actions[0] if len(actions) > 0 else 3,
                "actions_str": ";".join(map(str, actions)),
            })

            # Fixed-4000 per preset
            r_4k = pu.run_pipeline(img1, img2, preset, 4000, key1=key1, key2=key2)
            m_4k = evaluate_matches(r_4k["pts1"], r_4k["pts2"], H, img1.shape, ransac_seed=get_ransac_seed(seq, k, f"{preset}_4k"))
            rows.append({
                "fold": args.fold, "split": args.split, "seq": seq, "k": k,
                "method": f"{preset}+Fixed-4000", "preset": preset,
                "final_kp": float(r_4k["n_kp"]), "cum_kp": float(r_4k["n_kp"]), "steps": 0,
                "corner_err": float(m_4k["corner_err"]), "success": float(m_4k["h_ok"]),
                "first_action": -1, "actions_str": "",
            })

        # SIFT Fixed budgets: {500, 1000, 1500, 2000, 3000, 4000}
        for b in [500, 1000, 1500, 2000, 3000, 4000]:
            r_sift = pu.run_pipeline(img1, img2, "SIFT", b, key1=key1, key2=key2)
            m_sift = evaluate_matches(r_sift["pts1"], r_sift["pts2"], H, img1.shape, ransac_seed=get_ransac_seed(seq, k, f"SIFT_{b}"))
            rows.append({
                "fold": args.fold, "split": args.split, "seq": seq, "k": k,
                "method": f"SIFT Fixed-{b}", "preset": "SIFT",
                "final_kp": float(r_sift["n_kp"]), "cum_kp": float(r_sift["n_kp"]), "steps": 0,
                "corner_err": float(m_sift["corner_err"]), "success": float(m_sift["h_ok"]),
                "first_action": -1, "actions_str": "",
            })

        # Learned Baselines: ALIKED-500, XFeat-1000
        r_al = pu.run_pipeline(img1, img2, "ALIKED", 500, key1=key1, key2=key2)
        m_al = evaluate_matches(r_al["pts1"], r_al["pts2"], H, img1.shape, ransac_seed=get_ransac_seed(seq, k, "ALIKED_500"))
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "ALIKED-500", "preset": "ALIKED",
            "final_kp": float(r_al["n_kp"]), "cum_kp": float(r_al["n_kp"]), "steps": 0,
            "corner_err": float(m_al["corner_err"]), "success": float(m_al["h_ok"]),
            "first_action": -1, "actions_str": "",
        })

        r_xf = pu.run_pipeline(img1, img2, "XFeat", 1000, key1=key1, key2=key2)
        m_xf = evaluate_matches(r_xf["pts1"], r_xf["pts2"], H, img1.shape, ransac_seed=get_ransac_seed(seq, k, "XFeat_1000"))
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "XFeat-1000", "preset": "XFeat",
            "final_kp": float(r_xf["n_kp"]), "cum_kp": float(r_xf["n_kp"]), "steps": 0,
            "corner_err": float(m_xf["corner_err"]), "success": float(m_xf["h_ok"]),
            "first_action": -1, "actions_str": "",
        })

        # P1: Best Fixed (Fold VAL)
        # Folds 0, 1 -> ALIKED-4000; Folds 2, 3, 4 -> SIFT-4000
        val_best_preset = "ALIKED" if args.fold in [0, 1] else "SIFT"
        val_best_budget = 4000
        r_val_best = pu.run_pipeline(img1, img2, val_best_preset, val_best_budget, key1=key1, key2=key2)
        m_val_best = evaluate_matches(r_val_best["pts1"], r_val_best["pts2"], H, img1.shape, ransac_seed=get_ransac_seed(seq, k, f"{val_best_preset}_{val_best_budget}"))
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "Best Fixed (Fold VAL)", "preset": val_best_preset,
            "final_kp": float(r_val_best["n_kp"]), "cum_kp": float(r_val_best["n_kp"]), "steps": 0,
            "corner_err": float(m_val_best["corner_err"]), "success": float(m_val_best["h_ok"]),
            "first_action": -1, "actions_str": "",
        })

        # Full AFM-RL (Level 1 Choice)
        _, chosen_p = l1_selections[(seq, k)]
        chosen_rollout = pair_preset_rollouts[chosen_p]
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "Full AFM-RL (Strict STOP)", "preset": chosen_p,
            "final_kp": chosen_rollout["step_info"]["final_n_kp"], "cum_kp": chosen_rollout["cum_kp"],
            "steps": chosen_rollout["steps"], "corner_err": chosen_rollout["step_info"]["final_corner_err"],
            "success": chosen_rollout["step_info"]["final_h_ok"],
            "first_action": chosen_rollout["first_action"],
            "actions_str": ";".join(map(str, chosen_rollout["actions"])),
        })
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "Full AFM-RL (Proxy Scoreboard)", "preset": chosen_p,
            "final_kp": chosen_rollout["step_info"]["best_n_kp"], "cum_kp": chosen_rollout["cum_kp"],
            "steps": chosen_rollout["steps"], "corner_err": chosen_rollout["step_info"]["best_corner_err"],
            "success": chosen_rollout["step_info"]["best_h_ok"],
            "first_action": chosen_rollout["first_action"],
            "actions_str": ";".join(map(str, chosen_rollout["actions"])),
        })

        # Ablations: L1 Preset + Fixed-500 and Fixed-1000 (No L2)
        r_l1_500 = pu.run_pipeline(img1, img2, chosen_p, 500, key1=key1, key2=key2)
        m_l1_500 = evaluate_matches(r_l1_500["pts1"], r_l1_500["pts2"], H, img1.shape, ransac_seed=get_ransac_seed(seq, k, f"{chosen_p}_500"))
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "L1 Preset + Fixed-500", "preset": chosen_p,
            "final_kp": float(r_l1_500["n_kp"]), "cum_kp": float(r_l1_500["n_kp"]), "steps": 0,
            "corner_err": float(m_l1_500["corner_err"]), "success": float(m_l1_500["h_ok"]),
            "first_action": -1, "actions_str": "",
        })

        r_l1_1k = pu.run_pipeline(img1, img2, chosen_p, 1000, key1=key1, key2=key2)
        m_l1_1k = evaluate_matches(r_l1_1k["pts1"], r_l1_1k["pts2"], H, img1.shape, ransac_seed=get_ransac_seed(seq, k, f"{chosen_p}_1000"))
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "L1 Preset + Fixed-1000", "preset": chosen_p,
            "final_kp": float(r_l1_1k["n_kp"]), "cum_kp": float(r_l1_1k["n_kp"]), "steps": 0,
            "corner_err": float(m_l1_1k["corner_err"]), "success": float(m_l1_1k["h_ok"]),
            "first_action": -1, "actions_str": "",
        })

        # SIFT+L2
        sift_rollout = pair_preset_rollouts["SIFT"]
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "SIFT+L2 (Strict STOP)", "preset": "SIFT",
            "final_kp": sift_rollout["step_info"]["final_n_kp"], "cum_kp": sift_rollout["cum_kp"],
            "steps": sift_rollout["steps"], "corner_err": sift_rollout["step_info"]["final_corner_err"],
            "success": sift_rollout["step_info"]["final_h_ok"],
            "first_action": sift_rollout["first_action"],
            "actions_str": ";".join(map(str, sift_rollout["actions"])),
        })
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "SIFT+L2 (Proxy Scoreboard)", "preset": "SIFT",
            "final_kp": sift_rollout["step_info"]["best_n_kp"], "cum_kp": sift_rollout["cum_kp"],
            "steps": sift_rollout["steps"], "corner_err": sift_rollout["step_info"]["best_corner_err"],
            "success": sift_rollout["step_info"]["best_h_ok"],
            "first_action": sift_rollout["first_action"],
            "actions_str": ";".join(map(str, sift_rollout["actions"])),
        })

        # Oracle Preset (GT chosen minimum corner error under L2)
        oracle_p = min(pair_preset_rollouts.keys(), key=lambda p: pair_preset_rollouts[p]["step_info"]["final_corner_err"])
        oracle_rollout = pair_preset_rollouts[oracle_p]
        rows.append({
            "fold": args.fold, "split": args.split, "seq": seq, "k": k,
            "method": "Oracle Preset Selection", "preset": oracle_p,
            "final_kp": oracle_rollout["step_info"]["final_n_kp"], "cum_kp": oracle_rollout["cum_kp"],
            "steps": oracle_rollout["steps"], "corner_err": oracle_rollout["step_info"]["final_corner_err"],
            "success": oracle_rollout["step_info"]["final_h_ok"],
            "first_action": oracle_rollout["first_action"],
            "actions_str": ";".join(map(str, oracle_rollout["actions"])),
        })

    df_new = pd.DataFrame(rows)

    if out_path.exists() and not args.force:
        existing_df = pd.read_csv(out_path)
        combined_df = pd.concat([existing_df, df_new], ignore_index=True)
        combined_df.to_csv(out_path, index=False)
        print(f"Appended {len(df_new)} rows to {out_path} (Total: {len(combined_df)})")
    else:
        df_new.to_csv(out_path, index=False)
        print(f"Saved {len(df_new)} rows to {out_path}")

    return df_new

if __name__ == "__main__":
    args = parse_args()
    run_benchmark_fold(args)

