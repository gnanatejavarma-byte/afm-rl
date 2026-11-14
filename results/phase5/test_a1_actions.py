import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from level2_env import Level2Env, STEP_SIZES as L2_STEP_SIZES
from megadepth_eval_env import MegaDepthEvalEnv
import pipeline_utils as pu

# Direct env calls test
records = []

# Test Level2Env
l2_env = Level2Env(split="val", obs_mode="proxy", eval_mode=True)
l2_env.reset_to("i_ajuntament", 2, "SIFT")

for N_val in [100, 500, 4000]:
    for a in range(4):
        # Reset state to specific N
        l2_env.n_kp = N_val
        l2_env.steps = 0
        l2_env._in_eval_rollout = True
        obs, reward, terminated, truncated, info = l2_env.step(a)
        records.append({
            "env": "Level2Env (HPatches Training/Eval)",
            "action": a,
            "semantic_intent": "STOP" if a == 3 else f"{L2_STEP_SIZES[a]:+d}",
            "N_before": N_val,
            "N_after": l2_env.n_kp,
            "terminated": terminated,
            "truncated": truncated
        })

# Test MegaDepthEvalEnv
md_env = MegaDepthEvalEnv(eval_mode=True)
md_env.reset_to("0015_0.1_0.3.npz", 580, 1398, "SIFT")

for N_val in [100, 500, 4000]:
    for a in range(4):
        md_env.n_kp = N_val
        md_env.steps = 0
        md_env._in_eval_rollout = True
        # semantic intent in md_env:
        md_semantics = {0: "STOP", 1: "+500", 2: "-500", 3: "+1000"}
        obs, reward, terminated, truncated, info = md_env.step(a)
        records.append({
            "env": "MegaDepthEvalEnv (MegaDepth Eval)",
            "action": a,
            "semantic_intent": md_semantics[a],
            "N_before": N_val,
            "N_after": md_env.n_kp,
            "terminated": terminated,
            "truncated": truncated
        })

df = pd.DataFrame(records)
df.to_csv("results/phase5/a1_env_action_comparison.csv", index=False)
print("=== A.1 DIRECT ENV CALLS COMPARISON TABLE ===")
print(df.to_string(index=False))
