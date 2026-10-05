import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from stable_baselines3 import PPO
from level2_env import Level2Env

model = PPO.load("checkpoints/level2_best/best_model.zip")
env = Level2Env(split="val", seed=11)

max_ns = []
for _ in range(100):
    obs, info = env.reset()
    peak = info.get("n_kp", 500) if isinstance(info, dict) and "n_kp" in info else 500
    peak = env.n_kp
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        peak = max(peak, info["n_kp"])
        if terminated or truncated:
            break
    max_ns.append(peak)

max_ns = np.array(max_ns)
print(f"Episodes reaching N >= 2000: {(max_ns >= 2000).sum()} / {len(max_ns)}")
print(f"Episodes reaching N >= 3000: {(max_ns >= 3000).sum()} / {len(max_ns)}")
print(f"Episodes reaching N = 4000:  {(max_ns == 4000).sum()} / {len(max_ns)}")
print(f"Mean / max peak N reached: {max_ns.mean():.0f} / {max_ns.max()}")