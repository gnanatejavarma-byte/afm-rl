import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO

import pipeline_utils as pu
from metrics import evaluate_matches, compute_label_free_proxy
from hpatches import load_split, load_pair, list_pairs
from level1_env import Level1Env
from level2_env import Level2Env, STEP_SIZES, MAX_STEPS, STEP_PENALTY
from image_stats import compute_pair_stats, normalize_stats

L1_MODEL_PATH = "checkpoints/level1_best/best_model.zip"
L2_MODEL_PATH = "checkpoints/level2_14preset_best/best_model.zip"

def auc(errors, tau, n=400):
    e = np.clip(np.nan_to_num(np.asarray(errors, dtype=float), nan=1e6, posinf=1e6), 0, tau)
    ts = np.linspace(0, tau, n)
    recall = np.array([(e < t).mean() for t in ts])
    return float((np.trapezoid(recall, ts) if hasattr(np, 'trapezoid') else np.trapz(recall, ts)) / tau)

# ---------------------------------------------------------------------------
# Part 1: Train a Linear / Ridge Classifier on Level 1 training data (Pure NumPy)
# ---------------------------------------------------------------------------
def train_linear_classifier():
    split = load_split()
    train_pairs = list_pairs(split["train"])
    print(f"Collecting training statistics from {len(train_pairs)} training pairs...")
    X, Y = [], []
    for seq, k in train_pairs[:200]:  # sample 200 pairs for fast fitting
        img1, img2, H = load_pair(seq, k)
        stats = normalize_stats(compute_pair_stats(img1, img2))
        # Best preset on this pair (oracle from pool)
        best_p_idx = 1  # default SIFT
        best_err = float("inf")
        for idx, p in enumerate(["SIFT", "ALIKED", "ORB", "AKAZE", "FAST+BRIEF"]):
            r = pu.run_pipeline(img1, img2, p, 1000, key1=(seq, 1), key2=(seq, k))
            m = evaluate_matches(r["pts1"], r["pts2"], H, img1.shape)
            if m["corner_err"] < best_err:
                best_err = m["corner_err"]
                best_p_idx = pu.PRESET_NAMES.index(p)
        X.append(stats)
        y_onehot = np.zeros(len(pu.PRESET_NAMES))
        y_onehot[best_p_idx] = 1.0
        Y.append(y_onehot)

    X = np.array(X)
    Y = np.array(Y)
    # Ridge regression: W = (X^T X + alpha I)^(-1) X^T Y
    alpha = 1.0
    W = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y
    return W

# ---------------------------------------------------------------------------
# Part 2: Ablation Experiments on Validation Split (90 pairs)
# ---------------------------------------------------------------------------
def run_ablations():
    print("=" * 70)
    print("RUNNING ABLATION STUDIES ON HPATCHES VALIDATION SPLIT (90 PAIRS)")
    print("=" * 70)

    val_pairs = list_pairs(load_split()["val"])
    l1_model = PPO.load(L1_MODEL_PATH)
    l2_model = PPO.load(L2_MODEL_PATH)
    W_linear = train_linear_classifier()

    # Pre-load pairs to save I/O
    pairs_data = []
    for seq, k in val_pairs:
        img1, img2, H = load_pair(seq, k)
        stats = normalize_stats(compute_pair_stats(img1, img2))
        pairs_data.append((seq, k, img1, img2, H, stats))

    ablation_configs = [
        {"name": "Full AFM-RL (Default: lam=0.10, c_step=0.01, L1-Bandit, Strict-STOP)", "lam": 0.10, "c_step": 0.01, "l1": "bandit", "rule": "strict_stop"},
        {"name": "Ablation: Cost Penalty lam = 0.0 (Accuracy Only, No Keypoint Penalty)", "lam": 0.00, "c_step": 0.01, "l1": "bandit", "rule": "strict_stop"},
        {"name": "Ablation: Cost Penalty lam = 0.05 (Low Penalty)", "lam": 0.05, "c_step": 0.01, "l1": "bandit", "rule": "strict_stop"},
        {"name": "Ablation: Cost Penalty lam = 0.20 (High Penalty)", "lam": 0.20, "c_step": 0.01, "l1": "bandit", "rule": "strict_stop"},
        {"name": "Ablation: Step Penalty c_step = 0.0 (No Step Tax)", "lam": 0.10, "c_step": 0.00, "l1": "bandit", "rule": "strict_stop"},
        {"name": "Ablation: Step Penalty c_step = 0.05 (Aggressive Step Tax)", "lam": 0.10, "c_step": 0.05, "l1": "bandit", "rule": "strict_stop"},
        {"name": "Ablation: Level 1 Classifier (Linear / Ridge Regression on 9 stats)", "lam": 0.10, "c_step": 0.01, "l1": "linear", "rule": "strict_stop"},
        {"name": "Ablation: Level 1 Random Selector (Uniform Random Preset)", "lam": 0.10, "c_step": 0.01, "l1": "random", "rule": "strict_stop"},
        {"name": "Ablation: Fixed SIFT Preset (No Level 1 Selection)", "lam": 0.10, "c_step": 0.01, "l1": "sift", "rule": "strict_stop"},
        {"name": "Ablation: Decision Rule — Proxy Scoreboard", "lam": 0.10, "c_step": 0.01, "l1": "bandit", "rule": "proxy_scoreboard"},
        {"name": "Ablation: Decision Rule — Oracle Scoreboard (Upper Bound)", "lam": 0.10, "c_step": 0.01, "l1": "bandit", "rule": "oracle_scoreboard"},
    ]

    ablation_results = []

    for cfg in ablation_configs:
        print(f"Running config: {cfg['name']}...")
        env = Level2Env(split="val", lam=cfg["lam"], seed=42, eval_mode=True)
        errors = []
        keypoints = []
        steps_list = []
        t0 = time.perf_counter()

        for seq, k, img1, img2, H, stats in pairs_data:
            # 1. Preset selection
            if cfg["l1"] == "bandit":
                act, _ = l1_model.predict(stats, deterministic=True)
                preset = pu.PRESET_NAMES[int(act)]
            elif cfg["l1"] == "linear":
                scores = stats @ W_linear
                preset = pu.PRESET_NAMES[int(np.argmax(scores))]
            elif cfg["l1"] == "random":
                preset = np.random.choice(pu.PRESET_NAMES)
            elif cfg["l1"] == "sift":
                preset = "SIFT"

            # 2. Level 2 rollout
            obs, info = env.reset_to(seq, k, preset)
            step_count = 0
            while True:
                act, _ = l2_model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(int(act))
                step_count += 1
                if term or trunc:
                    break

            # 3. Apply Decision Rule
            if cfg["rule"] == "strict_stop":
                err = info["final_corner_err"]
                kp = info["final_n_kp"]
            elif cfg["rule"] == "proxy_scoreboard":
                err = info["best_corner_err"]
                kp = info["best_n_kp"]
            elif cfg["rule"] == "oracle_scoreboard":
                err = min(info["final_corner_err"], info["best_corner_err"])
                kp = info["best_n_kp"]

            errors.append(err)
            keypoints.append(kp)
            steps_list.append(step_count)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(pairs_data)
        succ3 = (np.array(errors) < 3.0).mean() * 100.0
        auc3 = auc(errors, 3.0) * 100.0
        auc5 = auc(errors, 5.0) * 100.0
        avg_kp = float(np.mean(keypoints))
        avg_steps = float(np.mean(steps_list))

        ablation_results.append({
            "Configuration": cfg["name"],
            "Avg Keypoints": round(avg_kp, 1),
            "Avg Steps": round(avg_steps, 2),
            "Latency (ms)": round(elapsed_ms, 2),
            "Success@3px (%)": round(succ3, 2),
            "AUC@3px (%)": round(auc3, 2),
            "AUC@5px (%)": round(auc5, 2),
        })

    df_ablations = pd.DataFrame(ablation_results)
    df_ablations.to_csv("logs/ablation_study_val90.csv", index=False)
    print("\nSaved logs/ablation_study_val90.csv")
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS (HPATCHES VALIDATION SPLIT, 90 PAIRS)")
    print("=" * 90)
    print(df_ablations.to_string(index=False))
    print("=" * 90)

# ---------------------------------------------------------------------------
# Part 3: Statistical Rigor & Multi-Seed Evaluation (M5)
# ---------------------------------------------------------------------------
def run_multi_seed_evaluation():
    print("\n" + "=" * 70)
    print("RUNNING MULTI-SEED EVALUATION ON HPATCHES TEST SPLIT (85 PAIRS)")
    print("=" * 70)

    test_pairs = list_pairs(load_split()["test"])
    l1_model = PPO.load(L1_MODEL_PATH)
    l2_model = PPO.load(L2_MODEL_PATH)

    pairs_data = []
    for seq, k in test_pairs:
        img1, img2, H = load_pair(seq, k)
        stats = normalize_stats(compute_pair_stats(img1, img2))
        pairs_data.append((seq, k, img1, img2, H, stats))

    seeds = [42, 100, 123, 777, 999]
    seed_records = []

    for seed in seeds:
        np.random.seed(seed)
        env = Level2Env(split="test", seed=seed, eval_mode=True)
        errors = []
        keypoints = []
        times = []

        for seq, k, img1, img2, H, stats in pairs_data:
            t0 = time.perf_counter()
            act, _ = l1_model.predict(stats, deterministic=True)
            preset = pu.PRESET_NAMES[int(act)]

            obs, info = env.reset_to(seq, k, preset)
            while True:
                a, _ = l2_model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(int(a))
                if term or trunc:
                    break
            t_ms = (time.perf_counter() - t0) * 1000.0

            errors.append(info["final_corner_err"])
            keypoints.append(info["final_n_kp"])
            times.append(t_ms)

        s3 = (np.array(errors) < 3.0).mean() * 100.0
        a1 = auc(errors, 1.0) * 100.0
        a3 = auc(errors, 3.0) * 100.0
        a5 = auc(errors, 5.0) * 100.0
        kp = float(np.mean(keypoints))
        lat = float(np.mean(times))

        seed_records.append({
            "Seed": seed,
            "Success@3px": s3,
            "AUC@1px": a1,
            "AUC@3px": a3,
            "AUC@5px": a5,
            "Avg_Keypoints": kp,
            "Latency_ms": lat
        })
        print(f"Seed {seed:3d}: Success@3px={s3:.2f}%, AUC@3px={a3:.2f}%, Keypoints={kp:.1f}, Latency={lat:.1f}ms")

    df_seeds = pd.DataFrame(seed_records)
    df_seeds.to_csv("logs/multi_seed_statistics.csv", index=False)
    print("\nSaved logs/multi_seed_statistics.csv")

    summary_stats = []
    for col in ["Success@3px", "AUC@1px", "AUC@3px", "AUC@5px", "Avg_Keypoints", "Latency_ms"]:
        vals = df_seeds[col].values
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95 = 1.96 * (std / np.sqrt(len(vals)))
        summary_stats.append({
            "Metric": col,
            "Mean": round(mean, 2),
            "Std Dev": round(std, 2),
            "95% Conf Interval": f"±{ci95:.2f}",
            "Report String": f"{mean:.2f} ± {std:.2f}"
        })

    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv("logs/multi_seed_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("STATISTICAL RIGOR SUMMARY (5 EVALUATION SEEDS)")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    run_ablations()
    run_multi_seed_evaluation()

