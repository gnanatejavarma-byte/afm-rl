"""Training Level-2 RL Policy on MegaDepth-1500.

Trains a PPO agent specifically on the 1,050 MegaDepth training pairs,
evaluating periodically on the 225 validation pairs.
The 225 test pairs are strictly held out.

Checkpoints: experiments/megadepth_training/checkpoints/
Logs:        experiments/megadepth_training/logs/
"""

import argparse
from pathlib import Path
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
MEGADEPTH_EXP = PROJECT_ROOT / "experiments" / "megadepth"
EXP_DIR = Path(__file__).resolve().parent

for p in [str(SRC_DIR), str(MEGADEPTH_EXP), str(EXP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from megadepth_level2_env import MegaDepthLevel2Env

SEED = 42
TOTAL_TIMESTEPS = 170_528
CHECKPOINT_FREQ = 10_000
EVAL_FREQ = 10_000


def train(smoke_test: bool = False, resume: bool = False, timesteps: int | None = None):
    ckpt_dir = EXP_DIR / "checkpoints"
    log_dir = EXP_DIR / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = 256 if smoke_test else (timesteps or TOTAL_TIMESTEPS)
    n_steps = 128 if smoke_test else 1024
    batch_size = 64 if smoke_test else 128
    eval_freq = 128 if smoke_test else EVAL_FREQ
    n_eval_episodes = 5 if smoke_test else 40

    print("=" * 75)
    print("MegaDepth-1500 Level-2 RL Training (PPO)")
    print(f"Mode: {'SMOKE TEST' if smoke_test else 'FULL TRAINING'}")
    print(f"Target Total Timesteps: {total_timesteps:,} | n_steps: {n_steps} | batch_size: {batch_size}")
    print(f"Checkpoint Dir: {ckpt_dir}")
    print(f"Log Dir:        {log_dir}")
    print("=" * 75)

    train_env = Monitor(MegaDepthLevel2Env(split="train", seed=SEED))
    val_env = Monitor(MegaDepthLevel2Env(split="val", seed=SEED + 1))

    load_path = None
    if resume:
        # Find checkpoint with highest num_timesteps
        best_ts = -1
        for f in ckpt_dir.glob("*.zip"):
            try:
                temp_m = PPO.load(str(f))
                if temp_m.num_timesteps > best_ts:
                    best_ts = temp_m.num_timesteps
                    load_path = f
            except Exception:
                pass

    if load_path is not None:
        print(f"[Resume] Loading existing model from {load_path}")
        model = PPO.load(str(load_path), env=train_env, tensorboard_log=str(log_dir))
        print(f"         Starting from timestep {model.num_timesteps:,} -> target {total_timesteps:,}")
    else:
        print("[Init] Initializing fresh PPO policy (MlpPolicy)")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=SEED,
            tensorboard_log=str(log_dir),
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=3e-4,
            ent_coef=0.03,
            gamma=0.99,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=eval_freq if smoke_test else CHECKPOINT_FREQ,
        save_path=str(ckpt_dir),
        name_prefix="megadepth_l2_ppo",
    )

    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    print(f"\nRunning PPO learning to target {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="megadepth_l2",
        reset_num_timesteps=False if resume else True,
    )

    final_path = ckpt_dir / "megadepth_l2_final"
    model.save(str(final_path))
    print(f"\n[Done] Training complete.")
    print(f"       Final model saved -> {final_path}.zip (total steps: {model.num_timesteps:,})")
    print(f"       Best model saved  -> {best_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Level-2 RL on MegaDepth")
    parser.add_argument("--smoke-test", action="store_true", help="Run small smoke test")
    parser.add_argument("--full", action="store_true", help="Run full training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--timesteps", type=int, default=None, help="Custom timesteps")
    args = parser.parse_args()

    train(smoke_test=args.smoke_test, resume=args.resume, timesteps=args.timesteps)


if __name__ == "__main__":
    main()
