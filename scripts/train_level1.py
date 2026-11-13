import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from level1_env import Level1Env
import pipeline_utils as pu

def parse_args():
    parser = argparse.ArgumentParser(description="Train Level 1 Pipeline Selection Agent")
    parser.add_argument("--obs-mode", choices=["proxy", "gt"], default="proxy",
                        help="Observation mode for downstream Level 2: 'proxy' or 'gt'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="checkpoints_v2/seed42",
                        help="Output directory (loads Level 2 from here, saves Level 1 here)")
    parser.add_argument("--total-timesteps", type=int, default=30_000,
                        help="Total training timesteps")
    parser.add_argument("--l2-ckpt", type=str, default=None,
                        help="Explicit path to Level 2 checkpoint (overrides out-dir search)")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000,
                        help="Checkpoint save frequency")
    parser.add_argument("--eval-freq", type=int, default=2_000,
                        help="Evaluation frequency in timesteps")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to split JSON file (e.g. configs/folds/fold0.json)")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("AFM-RL: Training Level 1 Pipeline Selection Agent")
    print(f"Process PID:      {os.getpid()}")
    print(f"Observation Mode: {args.obs_mode}")
    print(f"Seed:             {args.seed}")
    print(f"Total Timesteps:  {args.total_timesteps}")
    print(f"Output Directory: {args.out_dir}")
    print(f"Candidate Presets ({len(pu.PRESET_NAMES)}): {pu.PRESET_NAMES}")
    print("=" * 60)

    # Determine Level 2 checkpoint path
    if args.l2_ckpt:
        l2_path = args.l2_ckpt
    else:
        candidate_1 = os.path.join(args.out_dir, "level2_14preset_best", "best_model.zip")
        candidate_2 = os.path.join(args.out_dir, "level2_best", "best_model.zip")
        if os.path.exists(candidate_1):
            l2_path = candidate_1
        elif os.path.exists(candidate_2):
            l2_path = candidate_2
        else:
            raise FileNotFoundError(
                f"Level 2 checkpoint not found in {args.out_dir}. "
                f"Checked: {candidate_1} and {candidate_2}. "
                f"Train Level 2 first or pass --l2-ckpt explicitly."
            )

    print(f"Loading frozen Level 2 evaluator from: {l2_path}")

    ckpt_dir = os.path.join(args.out_dir, "level1")
    best_dir = os.path.join(args.out_dir, "level1_best")
    log_dir = os.path.join(args.out_dir, "level1_tb")
    eval_log_dir = os.path.join(args.out_dir, "level1_eval")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    train_env = Monitor(Level1Env(split="train", seed=args.seed, l2_model_path=l2_path, obs_mode=args.obs_mode, split_file=args.split_file))
    eval_env = Monitor(Level1Env(split="val", seed=args.seed + 100, l2_model_path=l2_path, obs_mode=args.obs_mode, split_file=args.split_file))

    print("\nInitializing PPO Policy for Level 1 (Contextual Bandit)...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=log_dir,
        n_steps=256,
        batch_size=64,
        learning_rate=5e-4,
        ent_coef=0.04,     # Encourages exploring different presets
        gamma=0.0,         # Single-step contextual bandit (immediate reward)
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="level1_ppo",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=50,
        deterministic=True,
    )

    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="l1_run",
        reset_num_timesteps=True,
    )

    final_path = os.path.join(ckpt_dir, "level1_final")
    model.save(final_path)
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final model saved to: {final_path}.zip")
    print(f"Best model saved to:  {best_dir}/best_model.zip")
    print(f"TensorBoard logs:     {log_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
