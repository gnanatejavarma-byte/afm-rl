import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from level2_env import Level2Env

env = Level2Env(split="train", seed=1)
rng = np.random.default_rng(1)

N_EPISODES = 30
t0 = time.perf_counter()
total_steps = 0
for _ in range(N_EPISODES):
    obs, info = env.reset()
    while True:
        action = rng.integers(0, 5)
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated or truncated:
            break
elapsed = time.perf_counter() - t0

print(f"{N_EPISODES} episodes, {total_steps} total steps in {elapsed:.1f}s")
print(f"avg {elapsed/total_steps*1000:.1f} ms/step, avg {elapsed/N_EPISODES:.2f} s/episode")
print(f"estimated time for 10,000 episodes: {elapsed/N_EPISODES*10000/60:.1f} minutes")