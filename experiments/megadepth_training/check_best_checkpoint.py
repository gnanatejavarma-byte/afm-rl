import os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

EXP_DIR = Path("experiments/megadepth_training")
ckpt_dir = EXP_DIR / "checkpoints"
log_dir = EXP_DIR / "logs"

print("=== CHECKPOINTS ===")
for f in sorted(ckpt_dir.glob("*.zip")):
    try:
        m = PPO.load(str(f))
        print(f"{f.name:<40}: {m.num_timesteps:,} steps")
    except Exception as e:
        print(f"{f.name}: {e}")

print("\n=== EVALUATIONS ===")
eval_path = log_dir / "evaluations.npz"
if eval_path.exists():
    data = np.load(str(eval_path))
    timesteps = data["timesteps"]
    results = data["results"]
    means = results.mean(axis=1)
    stds = results.std(axis=1)
    for ts, m_r, s_r in zip(timesteps, means, stds):
        print(f"Timestep {ts:>7,}: Mean Reward = {m_r:>7.4f} +/- {s_r:.4f}")

    best_idx = int(np.argmax(means))
    print(f"\n---> BEST VALIDATION CHECKPOINT: Timestep {timesteps[best_idx]:,} with Mean Reward {means[best_idx]:.4f}")

    # Best model timestep in best_model.zip
    best_m = PPO.load(str(ckpt_dir / "best_model.zip"))
    print(f"---> best_model.zip recorded timesteps: {best_m.num_timesteps:,}")
