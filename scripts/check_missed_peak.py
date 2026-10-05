import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from stable_baselines3 import PPO
from level2_env import Level2Env

model = PPO.load("checkpoints/level2_deltaobs_best/best_model.zip")
env = Level2Env(split="val", seed=11)

missed = []   # (preset, n_at_peak, peak_utility, n_final, final_utility, gap)
for _ in range(150):
    obs, info = env.reset()
    preset = env.preset
    trajectory = [(env.n_kp, env.utility)]   # (N, utility) at every point visited
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        trajectory.append((env.n_kp, env.utility))
        if terminated or truncated:
            break
    n_final, u_final = trajectory[-1]
    n_peak, u_peak = max(trajectory, key=lambda x: x[1])
    gap = u_peak - u_final
    if gap > 0.01 and n_peak != n_final:   # a REAL missed opportunity, not noise
        missed.append((preset, n_peak, round(u_peak, 3), n_final, round(u_final, 3), round(gap, 3)))

print(f"Episodes with a real missed peak: {len(missed)} / 150\n")
print(f"{'preset':12s} {'peak_N':>7s} {'peak_u':>7s} {'final_N':>8s} {'final_u':>8s} {'gap':>6s}")
for row in sorted(missed, key=lambda r: -r[5])[:20]:
    print(f"{row[0]:12s} {row[1]:7d} {row[2]:7.3f} {row[3]:8d} {row[4]:8.3f} {row[5]:6.3f}")