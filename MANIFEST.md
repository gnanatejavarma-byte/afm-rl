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
