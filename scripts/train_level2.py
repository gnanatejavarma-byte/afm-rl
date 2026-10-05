import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from level2_env import Level2Env

SEED = 42
TOTAL_TIMESTEPS = 150_000
CHECKPOINT_FREQ = 10_000
EVAL_FREQ = 10_000

LOG_DIR = "logs/level2_tb_deltaobs"
CKPT_DIR = "checkpoints/level2_deltaobs"
BEST_DIR = "checkpoints/level2_deltaobs_best"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

train_env = Monitor(Level2Env(split="train", seed=SEED))
eval_env = Monitor(Level2Env(split="val", seed=SEED + 1))

RESUME_FROM = "checkpoints/level2_deltaobs_best/best_model.zip"
if os.path.exists(RESUME_FROM):
    print(f"Resuming from {RESUME_FROM}")
    model = PPO.load(RESUME_FROM, env=train_env, tensorboard_log=LOG_DIR)
else:
    print("No existing 31-dim model found, training fresh")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
        n_steps=1024,
        batch_size=128,
        learning_rate=3e-4,
        ent_coef=0.03,
        gamma=0.99,
    )

checkpoint_cb = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CKPT_DIR,
    name_prefix="level2_ppo",
)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=BEST_DIR,
    log_path="logs/level2_eval_deltaobs",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=40,
    deterministic=True,
)

print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_cb, eval_cb],
            tb_log_name="run", reset_num_timesteps=False)

final_path = os.path.join(CKPT_DIR, "level2_final")
model.save(final_path)
print(f"Done. Final model saved to {final_path}.zip")
print(f"Best model (by val reward) saved to {BEST_DIR}/best_model.zip")