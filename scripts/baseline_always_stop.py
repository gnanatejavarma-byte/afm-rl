import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from level2_env import Level2Env

STOP_ACTION = 3   # index of None in STEP_SIZES [500, 1000, -500, None]

env = Level2Env(split="val", seed=99)
rewards = []
for _ in range(50):
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(STOP_ACTION)
    rewards.append(reward)

rewards = np.array(rewards)
print(f"Always-STOP baseline on val (N stays at start=500, 50 episodes):")
print(f"  mean reward = {rewards.mean():.3f}  (std {rewards.std():.3f})")