import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add project root and src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hpatches
import pipeline_utils as pu
from level2_env import Level2Env
from level1_env import Level1Env

def parse_args():
    parser = argparse.ArgumentParser(description="Run complete training pipeline for an AFM-RL fold (Level 2 + Level 1)")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0..4)")
    parser.add_argument("--split-file", type=str, default=None, help="Path to fold split JSON (defaults to configs/folds/fold{k}.json)")
    parser.add_argument("--out-dir", type=str, default=None, help="Base output directory for checkpoints (defaults to checkpoints_v2/fold{k})")
    parser.add_argument("--obs-mode", choices=["proxy", "gt"], default="proxy", help="Observation mode ('proxy' or 'gt')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to fold index)")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads for cv2 and torch")
    parser.add_argument("--l2-timesteps", type=int, default=150_000, help="Timesteps for Level 2 training")
    parser.add_argument("--l1-timesteps", type=int, default=30_000, help="Timesteps for Level 1 training")
    parser.add_argument("--l2-checkpoint-freq", type=int, default=10_000, help="Checkpoint frequency for Level 2")
    parser.add_argument("--l2-eval-freq", type=int, default=10_000, help="Evaluation frequency for Level 2")
    parser.add_argument("--l1-checkpoint-freq", type=int, default=5_000, help="Checkpoint frequency for Level 1")
    parser.add_argument("--l1-eval-freq", type=int, default=2_000, help="Evaluation frequency for Level 1")
    parser.add_argument("--skip-l2", action="store_true", default=False, help="Explicitly skip Level 2")
    parser.add_argument("--skip-l1", action="store_true", default=False, help="Explicitly skip Level 1")
    return parser.parse_args()

def assert_and_print_splits(split_file):
    print("=" * 70)
    print("SPLIT DISJOINTNESS & MODEL SELECTION AUDIT")
    print("=" * 70)
    print(f"Loading split file: {split_file}")
    s = hpatches.load_split(split_file)

    train_seqs = set(s["train"])
    val_seqs = set(s["val"])
    test_seqs = set(s["test"])
    all_seqs = train_seqs | val_seqs | test_seqs

    # 1. Print sequence counts
    print(f"Total sequences in split: {len(all_seqs)} (Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)})")

    # 2. Pairwise intersection sizes
    inter_train_val = len(train_seqs & val_seqs)
    inter_train_test = len(train_seqs & test_seqs)
    inter_val_test = len(val_seqs & test_seqs)

    print(f"Intersection Size | Train & Val:  {inter_train_val}")
    print(f"Intersection Size | Train & Test: {inter_train_test}")
    print(f"Intersection Size | Val & Test:   {inter_val_test}")

    # Assert strict pairwise disjointness
    assert inter_train_val == 0, f"LEAK DETECTED: Train and Val overlap by {inter_train_val} sequences!"
    assert inter_train_test == 0, f"LEAK DETECTED: Train and Test overlap by {inter_train_test} sequences!"
    assert inter_val_test == 0, f"LEAK DETECTED: Val and Test overlap by {inter_val_test} sequences!"

    # 3. Model selection audit
    print("\nModel Selection Policy Verification:")
    print("  -> Level 2 EvalCallback uses split: 'val' ONLY.")
    print("  -> Level 1 EvalCallback uses split: 'val' ONLY.")
    print("  -> TEST split is STRICTLY UNTOUCHED during all training and callbacks.")
    print("=" * 70 + "\n")

class Tee:
    def __init__(self, filename, original_stdout):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = original_stdout
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        self.stdout.flush()
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

def run_level2(args, out_dir, split_file, seed):
    log_file = os.path.join(ROOT, "results", "phase2c", f"fold{args.fold}_l2.log")
    tee = Tee(log_file, sys.stdout)
    orig_stdout = sys.stdout
    sys.stdout = tee
    try:
        return _run_level2_impl(args, out_dir, split_file, seed)
    finally:
        sys.stdout = orig_stdout
        tee.close()

def _run_level2_impl(args, out_dir, split_file, seed):
    l2_final = os.path.join(out_dir, "level2_14preset", "level2_14preset_final.zip")
    l2_best = os.path.join(out_dir, "level2_14preset_best", "best_model.zip")

    if os.path.exists(l2_final) and os.path.exists(l2_best):
        print(f"Stage 1 (Level 2) ALREADY COMPLETED. Found:\n  {l2_final}\n  {l2_best}\nSkipping Level 2 training.")
        return l2_best

    print("=" * 70)
    print(f"STARTING STAGE 1: Level 2 Training for Fold {args.fold}")
    print(f"  Obs Mode:        {args.obs_mode}")
    print(f"  Seed:            {seed}")
    print(f"  Total Timesteps: {args.l2_timesteps}")
    print(f"  Output Dir:      {out_dir}")
    print("=" * 70)

    ckpt_dir = os.path.join(out_dir, "level2_14preset")
    best_dir = os.path.join(out_dir, "level2_14preset_best")
    log_dir = os.path.join(out_dir, "tb")
    eval_log_dir = os.path.join(out_dir, "eval_logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    train_env = Monitor(Level2Env(split="train", seed=seed, obs_mode=args.obs_mode, eval_mode=False, split_file=split_file))
    eval_env = Monitor(Level2Env(split="val", seed=seed + 1, obs_mode=args.obs_mode, eval_mode=False, split_file=split_file))

    # Identical PPO hyperparameters as Seed 0
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=log_dir,
        n_steps=1024,
        batch_size=128,
        learning_rate=3e-4,
        ent_coef=0.03,
        gamma=0.99,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.l2_checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="level2_ppo",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=args.l2_eval_freq,
        n_eval_episodes=40,
        deterministic=True,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=args.l2_timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="run",
        reset_num_timesteps=True,
    )
    t_elapsed = time.time() - t0

    model.save(l2_final.replace(".zip", ""))
    print(f"\nStage 1 Complete! Final model: {l2_final} ({t_elapsed/60:.2f} min)")
    print(f"Best model saved to: {l2_best}")
    return l2_best

def run_level1(args, out_dir, split_file, seed, l2_best_path):
    log_file = os.path.join(ROOT, "results", "phase2c", f"fold{args.fold}_l1.log")
    tee = Tee(log_file, sys.stdout)
    orig_stdout = sys.stdout
    sys.stdout = tee
    try:
        return _run_level1_impl(args, out_dir, split_file, seed, l2_best_path)
    finally:
        sys.stdout = orig_stdout
        tee.close()

def _run_level1_impl(args, out_dir, split_file, seed, l2_best_path):
    l1_final = os.path.join(out_dir, "level1", "level1_final.zip")
    l1_best = os.path.join(out_dir, "level1_best", "best_model.zip")

    if os.path.exists(l1_final) and os.path.exists(l1_best):
        print(f"Stage 2 (Level 1) ALREADY COMPLETED. Found:\n  {l1_final}\n  {l1_best}\nSkipping Level 1 training.")
        return l1_best

    print("=" * 70)
    print(f"STARTING STAGE 2: Level 1 Training for Fold {args.fold}")
    print(f"  Obs Mode:        {args.obs_mode}")
    print(f"  Seed:            {seed}")
    print(f"  Total Timesteps: {args.l1_timesteps}")
    print(f"  Frozen L2 Model: {l2_best_path}")
    print(f"  Output Dir:      {out_dir}")
    print("=" * 70)

    ckpt_dir = os.path.join(out_dir, "level1")
    best_dir = os.path.join(out_dir, "level1_best")
    log_dir = os.path.join(out_dir, "level1_tb")
    eval_log_dir = os.path.join(out_dir, "level1_eval")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    train_env = Monitor(Level1Env(split="train", seed=seed, l2_model_path=l2_best_path, obs_mode=args.obs_mode, split_file=split_file))
    eval_env = Monitor(Level1Env(split="val", seed=seed + 100, l2_model_path=l2_best_path, obs_mode=args.obs_mode, split_file=split_file))

    # Identical PPO hyperparameters as Seed 0
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=log_dir,
        n_steps=256,
        batch_size=64,
        learning_rate=5e-4,
        ent_coef=0.04,
        gamma=0.0,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.l1_checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="level1_ppo",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=args.l1_eval_freq,
        n_eval_episodes=50,
        deterministic=True,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=args.l1_timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="l1_run",
        reset_num_timesteps=True,
    )
    t_elapsed = time.time() - t0

    model.save(l1_final.replace(".zip", ""))
    print(f"\nStage 2 Complete! Final model: {l1_final} ({t_elapsed/60:.2f} min)")
    print(f"Best model saved to: {l1_best}")
    return l1_best

def main():
    args = parse_args()

    # Set threads
    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)
    print(f"Process PID: {os.getpid()} | cv2 threads: {cv2.getNumThreads()} | torch threads: {torch.get_num_threads()}")

    # Determine split file and output directory
    split_file = args.split_file if args.split_file else f"configs/folds/fold{args.fold}.json"
    out_dir = args.out_dir if args.out_dir else f"checkpoints_v2/fold{args.fold}"
    seed = args.seed if args.seed is not None else args.fold

    # Step 1: Assertions
    assert_and_print_splits(split_file)

    # Step 2: Level 2 Training
    if not args.skip_l2:
        l2_best_path = run_level2(args, out_dir, split_file, seed)
    else:
        l2_best_path = os.path.join(out_dir, "level2_14preset_best", "best_model.zip")

    # Step 3: Level 1 Training
    if not args.skip_l1:
        run_level1(args, out_dir, split_file, seed, l2_best_path)

    print("\n" + "=" * 70)
    print(f"FOLD {args.fold} PIPELINE FULLY COMPLETED!")
    print(f"Checkpoints directory: {out_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
