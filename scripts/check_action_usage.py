import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from collections import Counter
from tqdm import tqdm
from stable_baselines3 import PPO
from level2_env import Level2Env

model = PPO.load("checkpoints/level2_deltaobs_best/best_model.zip")
env = Level2Env(split="val", seed=7)

N_EPISODES = 50
counts = Counter()
for _ in tqdm(range(N_EPISODES), desc="episodes"):
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        counts[int(action)] += 1
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break

labels = {0: "+500", 1: "+1000", 2: "-500", 3: "STOP"}
total = sum(counts.values())
for a in range(4):
    print(f"{labels[a]:6s}: {counts[a]:4d}  ({100*counts[a]/total:.1f}%)")