import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from level2_env import Level2Env

def parse_args():
    parser = argparse.ArgumentParser(description="Train Level 2 Feature Budget Agent")
    parser.add_argument("--obs-mode", choices=["proxy", "gt"], default="proxy",
                        help="Observation mode: 'proxy' (honest label-free) or 'gt' (legacy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="checkpoints_v2/seed42",
                        help="Output directory for checkpoints")
    parser.add_argument("--total-timesteps", type=int, default=150_000,
                        help="Total training timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000,
                        help="Checkpoint save frequency")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluation frequency in timesteps")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from existing checkpoint in out-dir if available")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to split JSON file (e.g. configs/folds/fold0.json)")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("AFM-RL: Training Level 2 Feature Budget Agent")
    print(f"Observation Mode: {args.obs_mode}")
    print(f"Seed:             {args.seed}")
    print(f"Total Timesteps:  {args.total_timesteps}")
    print(f"Output Directory: {args.out_dir}")
    print("=" * 60)

    ckpt_dir = os.path.join(args.out_dir, "level2_14preset")
    best_dir = os.path.join(args.out_dir, "level2_14preset_best")
    log_dir = os.path.join(args.out_dir, "tb")
    eval_log_dir = os.path.join(args.out_dir, "eval_logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    # Level 2 training and validation envs both use args.obs_mode
    # Reward comes from GT geometry; observations & scoreboard come from proxy
    train_env = Monitor(Level2Env(split="train", seed=args.seed, obs_mode=args.obs_mode, eval_mode=False, split_file=args.split_file))
    eval_env = Monitor(Level2Env(split="val", seed=args.seed + 1, obs_mode=args.obs_mode, eval_mode=False, split_file=args.split_file))

    resume_path = os.path.join(best_dir, "best_model.zip")
    if args.resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        model = PPO.load(resume_path, env=train_env, tensorboard_log=log_dir)
    else:
        print(f"Training fresh Level 2 model (obs_mode={args.obs_mode}, seed={args.seed})")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=log_dir,
            n_steps=1024,
            batch_size=128,
            learning_rate=3e-4,
            ent_coef=0.03,
            gamma=0.99,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="level2_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=40,
        deterministic=True,
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="run",
        reset_num_timesteps=not args.resume,
    )

    final_path = os.path.join(ckpt_dir, "level2_14preset_final")
    model.save(final_path)
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final model saved to: {final_path}.zip")
    print(f"Best model saved to:  {best_dir}/best_model.zip")
    print("=" * 60)

if __name__ == "__main__":
    main()
