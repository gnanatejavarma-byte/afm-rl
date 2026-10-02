import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from level2_env import Level2Env

SEED = 42
TOTAL_TIMESTEPS = 50_000
CHECKPOINT_FREQ = 5_000
EVAL_FREQ = 5_000

LOG_DIR = "logs/level2_tb"
CKPT_DIR = "checkpoints/level2"
BEST_DIR = "checkpoints/level2_best"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

train_env = Monitor(Level2Env(split="train", seed=SEED))
eval_env = Monitor(Level2Env(split="val", seed=SEED + 1))

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    seed=SEED,
    tensorboard_log=LOG_DIR,
    n_steps=1024,         # increase rollout buffer for broader cross-preset sampling
    batch_size=128,
    learning_rate=3e-4,
    ent_coef=0.03,        # enforce exploration across presets
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
    log_path="logs/level2_eval",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=20,
    deterministic=True,
)

print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_cb, eval_cb],
            tb_log_name="run")

final_path = os.path.join(CKPT_DIR, "level2_final")
model.save(final_path)
print(f"Done. Final model saved to {final_path}.zip")
print(f"Best model (by val reward) saved to {BEST_DIR}/best_model.zip")