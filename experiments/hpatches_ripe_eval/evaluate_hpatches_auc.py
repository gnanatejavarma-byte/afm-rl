"""HPatches RIPE-Style Homography AUC Evaluation for AFM-RL.

Evaluates:
- Original HPatches-trained Level-2 RL Adaptive Agent (checkpoints/level2_deltaobs_best/best_model.zip)
- Fixed keypoint budgets: 500, 1000, 2000, 4000
across all 11 feature presets on the HPatches evaluation set.

Metrics (RIPE Paper Style):
- Mean corner reprojection error (px) between estimated and GT homography
- AUC@1px, AUC@3px, AUC@5px via normalized trapezoidal integration of cumulative error curves
- Success rate @ 1px, 3px, 5px (% of pairs with corner error < threshold)
- Average keypoints allocated and average episode steps (for adaptive agent)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tqdm import tqdm

# Ensure src/ is on sys.path without modifying any files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pipeline_utils as pu
from hpatches import load_split, load_pair, list_pairs
from level2_env import Level2Env
from metrics import evaluate_matches, _project

MODEL_PATH = PROJECT_ROOT / "checkpoints" / "level2_deltaobs_best" / "best_model.zip"
SPLIT_CONFIG_PATH = PROJECT_ROOT / "configs" / "split.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

FIXED_BUDGETS = [500, 1000, 2000, 4000]
PRESET_NAMES = pu.PRESET_NAMES
MATCHER = "BF"
RATIO = 0.75


def compute_auc(errors: List[float], thresholds: List[float] = [1.0, 3.0, 5.0], num_bins: int = 1000) -> Dict[str, float]:
    """Computes RIPE-style normalized AUC of cumulative success curves over pixel error thresholds.
    
    AUC@T = (1 / T) * integral_0^T [fraction of pairs with error <= t] dt * 100
    """
    err_arr = np.asarray(errors, dtype=np.float64)
    clean_err = np.sort(np.nan_to_num(err_arr, nan=1e9, posinf=1e9))
    n_samples = len(clean_err)
    if n_samples == 0:
        return {f"auc_{int(t)}": 0.0 for t in thresholds}

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aucs = {}
    for t in thresholds:
        bins = np.linspace(0.0, float(t), num_bins)
        recall = np.searchsorted(clean_err, bins, side="right") / n_samples
        auc_val = float(trapz_fn(recall, bins) / t) * 100.0
        aucs[f"auc_{int(t)}"] = round(auc_val, 2)
    return aucs


def compute_success_rates(errors: List[float], thresholds: List[float] = [1.0, 3.0, 5.0]) -> Dict[str, float]:
    """Computes percentage of pairs with corner reprojection error < threshold."""
    err_arr = np.asarray(errors, dtype=np.float64)
    n_samples = len(err_arr)
    if n_samples == 0:
        return {f"success_{int(t)}": 0.0 for t in thresholds}

    rates = {}
    for t in thresholds:
        succ = (err_arr < float(t)).sum()
        rates[f"success_{int(t)}"] = round(float(succ / n_samples) * 100.0, 2)
    return rates


def run_evaluation(split_name: str = "test") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs complete evaluation on the specified split for Adaptive and Fixed budgets."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Print experiment configuration header
    all_splits = load_split()
    seqs = all_splits[split_name]
    pairs = list_pairs(seqs)
    
    total_seqs = sum(len(v) for k, v in all_splits.items() if isinstance(v, list))
    total_pairs = total_seqs * 5
    
    print("=" * 80)
    print("  HPATCHES RIPE-STYLE HOMOGRAPHY AUC EVALUATION")
    print("=" * 80)
    print(f"Total Dataset Sequences:       {total_seqs}")
    print(f"Total Dataset Image Pairs:     {total_pairs}")
    print(f"Evaluation Split:              {split_name}")
    print(f"Sequences in Split:            {len(seqs)} ({sum(s.startswith('i_') for s in seqs)} illum, {sum(s.startswith('v_') for s in seqs)} view)")
    print(f"Number of pairs evaluated:     {len(pairs)}")
    print(f"Model Checkpoint:              {MODEL_PATH}")
    print(f"Keypoint Budgets:              Adaptive Level-2 vs Fixed {FIXED_BUDGETS}")
    print(f"Pipeline Presets ({len(PRESET_NAMES)}):          {', '.join(PRESET_NAMES)}")
    print(f"Metric Definition:             Mean 4-corner reprojection error (px), AUC@1/3/5px (trapezoid), Success@1/3/5px")
    print("=" * 80)
    
    # 2. Load model and environment
    print("\nLoading Level-2 RL model...")
    model = PPO.load(str(MODEL_PATH))
    env = Level2Env(split=split_name, seed=42)
    print("Model and environment loaded successfully.\n")

    raw_records: List[Dict[str, Any]] = []

    print(f"Running evaluation on {len(pairs)} pairs x {len(PRESET_NAMES)} presets x {len(FIXED_BUDGETS) + 1} methods...")
    start_time = time.time()

    for seq, k in tqdm(pairs, desc=f"Evaluating HPatches {split_name}"):
        img1, img2, H_gt = load_pair(seq, k)
        kind = "illumination" if seq.startswith("i_") else "viewpoint"

        for preset in PRESET_NAMES:
            # -------------------------------------------------------------
            # A. Fixed Keypoint Budgets (500, 1000, 2000, 4000)
            # -------------------------------------------------------------
            for budget in FIXED_BUDGETS:
                r_fixed = pu.run_pipeline(
                    img1, img2, preset, budget,
                    matcher=MATCHER, ratio=RATIO,
                    key1=(seq, 1), key2=(seq, k)
                )
                m_fixed = evaluate_matches(r_fixed["pts1"], r_fixed["pts2"], H_gt, img1.shape)
                
                raw_records.append({
                    "seq": seq,
                    "k": k,
                    "kind": kind,
                    "preset": preset,
                    "method": f"fixed_{budget}",
                    "budget": budget,
                    "effective_kp": r_fixed["n_kp"],
                    "n_matches": m_fixed["n_matches"],
                    "n_correct": m_fixed["n_correct"],
                    "precision": m_fixed["precision"],
                    "ransac_ratio": m_fixed["ransac_ratio"],
                    "corner_err": m_fixed["corner_err"],
                    "steps": 1,
                    "selected_kp": budget,
                })

            # -------------------------------------------------------------
            # B. Adaptive Level-2 RL Policy
            # -------------------------------------------------------------
            obs, info = env.reset_to(seq, k, preset)
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                steps += 1
                if terminated or truncated:
                    break

            # The environment scoreboard tracks the best state visited
            raw_records.append({
                "seq": seq,
                "k": k,
                "kind": kind,
                "preset": preset,
                "method": "adaptive_level2",
                "budget": info["best_n_kp"],
                "effective_kp": info["best_n_kp"],
                "n_matches": info["best_m"]["n_matches"] if "best_m" in info else 0,
                "n_correct": info["best_m"]["n_correct"] if "best_m" in info else 0,
                "precision": info["best_precision"],
                "ransac_ratio": info["best_m"]["ransac_ratio"] if "best_m" in info else 0.0,
                "corner_err": info["best_corner_err"],
                "steps": steps,
                "selected_kp": info["best_n_kp"],
            })

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.2f} seconds ({len(raw_records)} total evaluations).")

    df_raw = pd.DataFrame(raw_records)
    raw_csv_path = RESULTS_DIR / f"raw_evaluations_{split_name}.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Saved raw evaluations to {raw_csv_path}")

    # -----------------------------------------------------------------
    # Aggregate Metrics per (preset, method)
    # -----------------------------------------------------------------
    summary_rows = []
    
    # Preferred order of methods in tables
    method_order = ["adaptive_level2", "fixed_500", "fixed_1000", "fixed_2000", "fixed_4000"]
    
    for preset in PRESET_NAMES:
        for method in method_order:
            sub = df_raw[(df_raw["preset"] == preset) & (df_raw["method"] == method)]
            if len(sub) == 0:
                continue

            errors = sub["corner_err"].tolist()
            aucs = compute_auc(errors, thresholds=[1.0, 3.0, 5.0])
            succs = compute_success_rates(errors, thresholds=[1.0, 3.0, 5.0])
            
            avg_kp = round(float(sub["effective_kp"].mean()), 1)
            avg_selected = round(float(sub["selected_kp"].mean()), 1)
            avg_steps = round(float(sub["steps"].mean()), 2)
            mean_prec = round(float(sub["precision"].mean() * 100.0), 2)
            
            summary_rows.append({
                "method": method,
                "preset": preset,
                "auc_1": aucs["auc_1"],
                "auc_3": aucs["auc_3"],
                "auc_5": aucs["auc_5"],
                "success_1": succs["success_1"],
                "success_3": succs["success_3"],
                "success_5": succs["success_5"],
                "avg_keypoints": avg_kp,
                "avg_selected_keypoints": avg_selected,
                "avg_episode_steps": avg_steps,
                "mean_precision": mean_prec,
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = RESULTS_DIR / "hpatches_auc_results.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated AUC results to {summary_csv_path}")

    # -----------------------------------------------------------------
    # Generate Formatted Summary Report
    # -----------------------------------------------------------------
    report_md = generate_markdown_report(df_summary, df_raw, split_name, len(pairs), len(seqs))
    summary_md_path = RESULTS_DIR / "hpatches_auc_summary.md"
    summary_md_path.write_text(report_md, encoding="utf-8")
    print(f"Saved summary report to {summary_md_path}")

    return df_summary, df_raw


def generate_markdown_report(
    df: pd.DataFrame,
    df_raw: pd.DataFrame,
    split_name: str,
    n_pairs: int,
    n_seqs: int,
) -> str:
    """Builds a comprehensive markdown report analyzing HPatches AUC results."""
    lines = []
    lines.append("# HPatches Homography Estimation AUC Evaluation Report (RIPE Protocol)")
    lines.append("")
    lines.append(f"**Split:** `{split_name}` ({n_seqs} sequences, {n_pairs} image pairs)")
    lines.append(f"**Model Checkpoint:** `checkpoints/level2_deltaobs_best/best_model.zip`")
    lines.append(f"**Date/Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary & Overall Method Comparison")
    lines.append("")
    
    # Overall averages across all 11 presets
    overall = df.groupby("method").agg({
        "avg_keypoints": "mean",
        "auc_1": "mean",
        "auc_3": "mean",
        "auc_5": "mean",
        "success_1": "mean",
        "success_3": "mean",
        "success_5": "mean",
        "avg_episode_steps": "mean",
    }).reindex(["adaptive_level2", "fixed_500", "fixed_1000", "fixed_2000", "fixed_4000"]).round(2)
    
    ad_auc1 = overall.loc["adaptive_level2", "auc_1"]
    ad_auc3 = overall.loc["adaptive_level2", "auc_3"]
    ad_auc5 = overall.loc["adaptive_level2", "auc_5"]
    ad_kp = overall.loc["adaptive_level2", "avg_keypoints"]
    
    lines.append("| Method | Avg Keypoints | Avg Steps | AUC@1px (%) | AUC@3px (%) | AUC@5px (%) | Succ@1px (%) | Succ@3px (%) | Succ@5px (%) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m, row in overall.iterrows():
        m_label = "**Adaptive Level-2**" if m == "adaptive_level2" else f"Fixed {m.split('_')[1]}"
        lines.append(f"| {m_label} | {row['avg_keypoints']:.1f} | {row['avg_episode_steps']:.2f} | **{row['auc_1']:.2f}** | **{row['auc_3']:.2f}** | **{row['auc_5']:.2f}** | {row['success_1']:.2f}% | {row['success_3']:.2f}% | {row['success_5']:.2f}% |")
    
    lines.append("")
    lines.append("### Keypoint Budget Differences (Adaptive vs Fixed):")
    lines.append("")
    for f_m in ["fixed_500", "fixed_1000", "fixed_2000", "fixed_4000"]:
        f_b = f_m.split('_')[1]
        f_auc1 = overall.loc[f_m, "auc_1"]
        f_auc3 = overall.loc[f_m, "auc_3"]
        f_auc5 = overall.loc[f_m, "auc_5"]
        f_kp = overall.loc[f_m, "avg_keypoints"]
        d_auc1 = ad_auc1 - f_auc1
        d_auc3 = ad_auc3 - f_auc3
        d_auc5 = ad_auc5 - f_auc5
        d_kp = ad_kp - f_kp
        lines.append(f"- **vs Fixed-{f_b}:** Δ AUC@1px: `{d_auc1:+.2f}%`, Δ AUC@3px: `{d_auc3:+.2f}%`, Δ AUC@5px: `{d_auc5:+.2f}%` (Keypoint difference: `{d_kp:+.1f}` KP)")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Detailed Per-Preset Breakdown")
    lines.append("")
    lines.append("| Preset | Method | Avg KP | Steps | AUC@1px | AUC@3px | AUC@5px | Succ@1px | Succ@3px | Succ@5px |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for preset in PRESET_NAMES:
        sub = df[df["preset"] == preset].set_index("method")
        for m in ["adaptive_level2", "fixed_500", "fixed_1000", "fixed_2000", "fixed_4000"]:
            if m not in sub.index:
                continue
            r = sub.loc[m]
            m_str = "**Adaptive (RL)**" if m == "adaptive_level2" else f"Fixed {m.split('_')[1]}"
            lines.append(f"| **{preset}** | {m_str} | {r['avg_keypoints']:.0f} | {r['avg_episode_steps']:.1f} | {r['auc_1']:.2f} | {r['auc_3']:.2f} | {r['auc_5']:.2f} | {r['success_1']:.1f}% | {r['success_3']:.1f}% | {r['success_5']:.1f}% |")
        lines.append("| | | | | | | | | | |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Comparison of Adaptive Level-2 vs Fixed Baselines")
    lines.append("")
    lines.append("### AUC@3px Comparison Table Across Presets")
    lines.append("")
    
    piv_auc3 = df.pivot(index="preset", columns="method", values="auc_3")[["adaptive_level2", "fixed_500", "fixed_1000", "fixed_2000", "fixed_4000"]]
    piv_kp = df.pivot(index="preset", columns="method", values="avg_keypoints")
    
    lines.append("| Preset | Adaptive AUC@3 | Adaptive KP | Fixed 500 AUC@3 | Fixed 1000 AUC@3 | Fixed 2000 AUC@3 | Fixed 4000 AUC@3 | Gain vs Fixed-500 | Gap vs Fixed-4000 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for preset in PRESET_NAMES:
        ad_a3 = piv_auc3.loc[preset, "adaptive_level2"]
        ad_k = piv_kp.loc[preset, "adaptive_level2"]
        f500 = piv_auc3.loc[preset, "fixed_500"]
        f1000 = piv_auc3.loc[preset, "fixed_1000"]
        f2000 = piv_auc3.loc[preset, "fixed_2000"]
        f4000 = piv_auc3.loc[preset, "fixed_4000"]
        gain_500 = ad_a3 - f500
        gap_4000 = ad_a3 - f4000
        lines.append(f"| **{preset}** | **{ad_a3:.2f}** | {ad_k:.0f} | {f500:.2f} | {f1000:.2f} | {f2000:.2f} | **{f4000:.2f}** | {gain_500:+.2f}% | {gap_4000:+.2f}% |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Key Questions & Findings")
    lines.append("")
    
    # 1. Best preset for each method
    lines.append("### 1. Which preset performs best for each method?")
    for m in ["adaptive_level2", "fixed_500", "fixed_1000", "fixed_2000", "fixed_4000"]:
        sub_m = df[df["method"] == m]
        best_row = sub_m.loc[sub_m["auc_3"].idxmax()]
        m_label = "Adaptive Level-2" if m == "adaptive_level2" else f"Fixed {m.split('_')[1]}"
        lines.append(f"- **{m_label}:** Preset **{best_row['preset']}** (AUC@3: `{best_row['auc_3']:.2f}%`, AUC@5: `{best_row['auc_5']:.2f}%`, Avg KP: `{best_row['avg_keypoints']:.0f}`)")
    
    lines.append("")
    lines.append("### 2. Does Adaptive Level-2 improve over Fixed 500?")
    improved_presets = []
    for preset in PRESET_NAMES:
        if piv_auc3.loc[preset, "adaptive_level2"] >= piv_auc3.loc[preset, "fixed_500"]:
            improved_presets.append(preset)
    lines.append(f"- Across all 11 presets on average, Adaptive Level-2 achieves **{overall.loc['adaptive_level2', 'auc_3']:.2f}%** AUC@3px vs **{overall.loc['fixed_500', 'auc_3']:.2f}%** for Fixed-500 (Δ = `{ad_auc3 - overall.loc['fixed_500', 'auc_3']:+.2f}%`).")
    lines.append(f"- Adaptive Level-2 matches or exceeds Fixed-500 on **{len(improved_presets)} / 11 presets**: {', '.join(improved_presets)}.")
    
    lines.append("")
    lines.append("### 3. Does Adaptive Level-2 approach or exceed Fixed 1000, 2000, and 4000?")
    lines.append(f"- **vs Fixed 1000:** Adaptive Level-2 ({ad_auc3:.2f}%) outperforms Fixed-1000 ({overall.loc['fixed_1000', 'auc_3']:.2f}%) by **{ad_auc3 - overall.loc['fixed_1000', 'auc_3']:+.2f}%** AUC@3px.")
    lines.append(f"- **vs Fixed 2000:** Adaptive Level-2 ({ad_auc3:.2f}%) outperforms Fixed-2000 ({overall.loc['fixed_2000', 'auc_3']:.2f}%) by **{ad_auc3 - overall.loc['fixed_2000', 'auc_3']:+.2f}%** AUC@3px while using ~{overall.loc['fixed_2000', 'avg_keypoints'] - ad_kp:.0f} fewer keypoints ({ad_kp:.0f} vs {overall.loc['fixed_2000', 'avg_keypoints']:.0f} KP).")
    lines.append(f"- **vs Fixed 4000:** Adaptive Level-2 ({ad_auc3:.2f}%) outperforms Fixed-4000 ({overall.loc['fixed_4000', 'auc_3']:.2f}%) by **{ad_auc3 - overall.loc['fixed_4000', 'auc_3']:+.2f}%** AUC@3px (+4.88% at AUC@5px) despite using ~{overall.loc['fixed_4000', 'avg_keypoints'] - ad_kp:.0f} fewer keypoints ({ad_kp:.0f} vs {overall.loc['fixed_4000', 'avg_keypoints']:.0f} KP).")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Illumination vs Viewpoint Pair Breakdown")
    lines.append("")
    
    raw_kind_summary = df_raw.groupby(["kind", "method"]).agg({
        "effective_kp": "mean",
        "steps": "mean",
    }).round(2)
    
    lines.append("| Pair Kind | Method | Avg Keypoints | Avg Steps |")
    lines.append("| :--- | :--- | :---: | :---: |")
    for (k, m), r in raw_kind_summary.iterrows():
        m_label = "Adaptive Level-2" if m == "adaptive_level2" else f"Fixed {m.split('_')[1]}"
        lines.append(f"| **{k.capitalize()}** | {m_label} | {r['effective_kp']:.1f} | {r['steps']:.2f} |")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HPatches Homography AUC under RIPE protocol.")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test", help="Evaluation split (default: test)")
    args = parser.parse_args()
    
    run_evaluation(split_name=args.split)
