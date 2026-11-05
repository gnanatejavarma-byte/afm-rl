import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from level1_env import Level1Env
import pipeline_utils as pu

SEED = 42
TOTAL_TIMESTEPS = 30_000
CHECKPOINT_FREQ = 5_000
EVAL_FREQ = 2_000

LOG_DIR  = "logs/level1_tb"
CKPT_DIR = "checkpoints/level1"
BEST_DIR = "checkpoints/level1_best"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

print("=" * 60)
print("AFM-RL: Training Level 1 Pipeline Selection Agent")
print(f"Candidate Presets ({len(pu.PRESET_NAMES)}): {pu.PRESET_NAMES}")
print("=" * 60)

train_env = Monitor(Level1Env(split="train", seed=SEED))
eval_env = Monitor(Level1Env(split="val", seed=SEED + 100))

print("\nInitializing PPO Policy for Level 1 (Contextual Bandit)...")
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    seed=SEED,
    tensorboard_log=LOG_DIR,
    n_steps=256,
    batch_size=64,
    learning_rate=5e-4,
    ent_coef=0.04,     # Encourages exploring different presets
    gamma=0.0,         # Single-step contextual bandit (immediate reward)
)

checkpoint_cb = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CKPT_DIR,
    name_prefix="level1_ppo",
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=BEST_DIR,
    log_path="logs/level1_eval",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=50,
    deterministic=True,
)

print(f"\nStarting training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_cb, eval_cb],
    tb_log_name="l1_run",
    reset_num_timesteps=True,
)

final_path = os.path.join(CKPT_DIR, "level1_final")
model.save(final_path)
print("\n" + "=" * 60)
print(f"Training Complete!")
print(f"Final model saved to: {final_path}.zip")
print(f"Best model saved to:  {BEST_DIR}/best_model.zip")
print(f"TensorBoard logs:     {LOG_DIR}")
print("=" * 60)

