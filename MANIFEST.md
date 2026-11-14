# Paper Reproduction Manifest: Tables, Figures, Scripts, and Source CSVs

This manifest maps every publication table and figure candidate to the exact script, raw source evaluation CSV, and processed summary CSV that produces it.

---

## 1. HPatches 5-Fold Cross-Validation Frozen Benchmark

### Table I: Main Benchmark Comparison (580 Held-Out Test Pairs)
* **Description:** Comprehensive comparison of fixed baselines, single-level policies, and full AFM-RL.
* **Evaluation Script:** `scripts/run_frozen_benchmark.py` (invoking `scripts/benchmark_folds.py`)
* **Aggregation Script:** `scripts/aggregate_folds.py --input results/phase3/folds_pairs.csv --out-dir results/phase3`
* **Raw Evaluation CSV:** `results/phase3/folds_pairs.csv` (25,520 rows)
* **Summary Output CSV:** `results/phase3/folds_summary_table.csv`
* **LaTeX Table:** `results/phase3/tables.tex`

### Table II: Primary Statistical Comparisons (P1 – P5)
* **Description:** Paired bootstrap differences (10,000 sequence clusters), 95% CIs, and exact two-sided McNemar tests.
* **Aggregation Script:** `scripts/aggregate_folds.py` (via `src/stats_tools.py`)
* **Raw Evaluation CSV:** `results/phase3/folds_pairs.csv`
* **Summary Output CSV:** `results/phase3/primary_comparisons_P1_P5.csv`
* **LaTeX Table:** `results/phase3/tables.tex` (`tab:phase3_primary_comparisons`)

### Table III: Component Ablations
* **Description:** Level 1 preset selection (Fixed-500, Fixed-1000) vs Level 2 dynamic budgeting (Strict STOP vs Proxy Scoreboard).
* **Aggregation Script:** `scripts/aggregate_folds.py`
* **Raw Evaluation CSV:** `results/phase3/folds_pairs.csv`
* **Summary Output CSV:** `results/phase3/ablation_table.csv`
* **LaTeX Table:** `results/phase3/tables.tex` (`tab:phase3_ablations`)

### Table IV: Hierarchical Policy Behavioral Audit
* **Description:** Level 1 preset distribution, Shannon entropy (bits), and Level 2 mean steps and STOP percentages per fold.
* **Aggregation Script:** `scripts/aggregate_folds.py`
* **Raw Evaluation CSV:** `results/phase3/folds_pairs.csv`
* **Summary Output CSVs:** `results/phase3/level1_selection_stats.csv`, `results/phase3/level2_action_stats.csv`
* **LaTeX Table:** `results/phase3/tables.tex` (`tab:phase3_behavior`)

### Figure / Dataset: Accuracy vs. Computation (Pareto Frontiers)
* **Description:** Success@3px and AUC@3px coordinates vs. Final and Cumulative Keypoints.
* **Aggregation Script:** `scripts/aggregate_folds.py`
* **Raw Evaluation CSV:** `results/phase3/folds_pairs.csv`
* **Dataset CSV:** `results/phase3/pareto_plot_data.csv`

---

## 2. Dev-Only Headroom, Noise Floor & Selector Diagnostics

### Diagnostic D1: RANSAC Noise Floor & Cross-Validated Oracle
* **Description:** 3-seed RANSAC evaluation (97,440 homographies), test-retest agreement, and cross-validated vs in-sample oracle.
* **Execution Scripts:** `results/phase3/dev_headroom_analysis.py`, `results/phase3/redo_dev_analyses.py`
* **Raw Evaluation CSV:** `results/phase3/dev_noise_floor_evaluations.csv` (97,440 rows)
* **Summary Output CSVs:** `results/phase3/dev_test_retest_agreement.csv`, `results/phase3/dev_oracle_noise_quantification.csv`

### Diagnostic D2: Label-Free Proxy Quality
* **Description:** Within-pair Spearman rank correlation ($\rho$), AUROC for registration success, and "Max Proxy" policy performance.
* **Execution Script:** `results/phase3/dev_headroom_analysis.py`
* **Raw Evaluation CSV:** `results/phase3/dev_noise_floor_evaluations.csv`
* **Summary Output CSV:** `results/phase3/dev_proxy_quality.csv`

### Diagnostic D3: Inner 5-Fold CV Learned Selectors (Train+Val Only)
* **Description:** Grouped cross-validation on train+val sequences (Logistic Regression, Gradient Boosting, Label-Free Cascade).
* **Execution Script:** `results/phase3/dev_headroom_analysis.py`
* **Raw Source Data:** `results/phase3/dev_noise_floor_evaluations.csv`, `configs/folds/fold{0..4}.json`
* **Summary Output CSV:** `results/phase3/dev_inner_cv_selectors.csv`

---

## 3. MegaDepth-1500 & Aachen Benchmark Tools

### MegaDepth 20-Pair Baseline Latency & Packaging Benchmark
* **Description:** Cold latency benchmarking on CPU for classical (SIFT+ratio-test) vs learned (ALIKED+MNN).
* **Execution Script:** `results/phase3b/time_megadepth_baselines.py`
* **Dataset Root:** `data/megadepth/megadepth_test_1500/`, `data/megadepth/megadepth_test_1500_scene_info/`

### Aachen Day-Night v1.1 Evaluation Pipeline
* **Description:** Visual localization recall evaluation on 225 pairs (150 Day + 75 Night).
* **Execution Script:** `scripts/eval_aachen_hierarchical.py`
* **Dataset Root:** `data/images_upright/`, `data/queries/`, `data/3D-models/`

---

## 4. Phase 5 MegaDepth Audit, Keypoint Accounting, Cascade Benchmark & Reproducibility

### Section A: AFM-RL v1 Failure Mechanism & Observation Shift Audit
* **Description:** Direct side-by-side verification of action semantics mismatch (STOP vs +1000) and 34-dim observation feature drift between HPatches training and MegaDepth evaluation.
* **Execution Scripts:** `results/phase5/test_a1_actions.py`, `results/phase5/run_section_a_diagnostics.py`
* **Source Modules:** `src/level2_env.py`, `src/megadepth_eval_env.py`
* **Summary Output CSVs:**
  * `results/phase5/a1_env_action_comparison.csv`
  * `results/phase5/a2_observation_shift_summary.csv`
  * `results/phase5/a3_rollout_diagnostics.csv`
  * `results/phase5/a3_native_semantics_test_100pairs.csv`

### Section B: Keypoint Accounting & Shortfall Audit
* **Description:** Comprehensive accounting of keypoint shortfalls (especially SIFT-4000) across scenes, overlap buckets, and detector presets.
* **Execution Script:** `results/phase5/b_keypoint_accounting.py`
* **Raw Source Data:** `results/phase4b/megadepth_stage2a_priority_evaluations.csv`, `results/phase4b/megadepth_stage2b_remaining_baselines.csv`
* **Summary Output CSV:** `results/phase5/b_keypoint_accounting_summary.csv`

### Section C: Cross-Scene Cascade Fitting & Primary Comparisons
* **Description:** Cross-scene parameter tuning (Train 0015 -> Test 0022 and vice-versa) and evaluation of pre-registered Cascade C* ($C_6$ `[ALIKED-500 -> SIFT-4000]`, $\tau=0.40$) vs fixed baselines.
* **Execution Script:** `results/phase5/c_cascade_cross_scene.py`
* **Raw Source Data:** `results/phase4b/megadepth_stage2a_priority_evaluations.csv`, `results/phase4b/megadepth_stage2b_remaining_baselines.csv`
* **Summary Output CSVs:**
  * `results/phase5/c_cross_scene_fitting_summary.csv`
  * `results/phase5/c_hpatches_preregistered_c_star.csv`
  * `results/phase5/c_primary_comparisons_summary.csv`

### Section D: Cascade Ablations & Sanity Controls
* **Description:** Ablation baselines including Always Both, Oracle-of-Two, Proxy-Pick ALIKED, Random Escalation Control, and Nearest Fixed baselines.
* **Execution Script:** `results/phase5/d_ablations.py`
* **Raw Source Data:** `results/phase4b/megadepth_stage2a_priority_evaluations.csv`, `results/phase4b/megadepth_stage2b_remaining_baselines.csv`
* **Summary Output CSV:** `results/phase5/d_ablations_summary.csv`

### Section E: Statistical Rigor & Correspondence Permutation Robustness
* **Description:** Pair-level and reference-image cluster bootstrap confidence intervals (1,000 resamples across 629 anchor image clusters) and 3-seed match permutation test.
* **Execution Scripts:** `results/phase5/e_statistical_evaluations.py`, `results/phase5/e_permutation_test.py`
* **Raw Source Data:** `results/phase4b/megadepth_stage2a_priority_evaluations.csv`, `results/phase4b/megadepth_stage2b_remaining_baselines.csv`, `results/phase4b/megadepth_matches/`
* **Summary Output CSVs:**
  * `results/phase5/e_statistical_comparisons.csv`
  * `results/phase5/e_permutation_robustness_summary.csv`

### Section F: CPU Wall-Clock Latency Benchmark
* **Description:** Cold-start latency breakdown across pair statistics, RL policy inference (L1 and L2), and feature extraction/matching on Intel CPU.
* **Execution Script:** `results/phase5/f_wallclock_benchmark.py`
* **Summary Output CSV:** `results/phase5/f_wallclock_summary.csv`

### Section G: Dataset Image & Pool File Reconciliation
* **Description:** Exact verification and reconciliation of physical images on disk, pair indices, and keypoint pool files across scenes.
* **Execution Script:** `results/phase5/g_housekeeping.py`
* **Summary Output CSV:** `results/phase5/g_image_reconciliation.csv`


