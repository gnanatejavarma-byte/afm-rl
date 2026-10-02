import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gymnasium.utils.env_checker import check_env
from level2_env import Level2Env

env = Level2Env(split="val", seed=0)
check_env(env, warn=True, skip_render_check=True)
print("check_env passed\n")

rng = np.random.default_rng(0)
for ep in range(3):
    obs, info = env.reset()
    total, steps = 0.0, 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            break
    print(f"episode {ep}: preset={info['preset']:11s} steps={steps:2d} "
          f"final_n_kp={info['n_kp']:5d} total_reward={total:6.2f} "
          f"final_precision={info['precision']:.2f} h_ok={info['h_ok']}")