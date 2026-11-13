# Phase 3 Reproduction Guide: Frozen Test Evaluation & Dev Analyses

This directory contains the protocol, aggregated test evaluation results, dev headroom diagnostics, and LaTeX tables for the honest, label-free AFM-RL benchmark across 5-fold cross-validation (580 test image pairs across 116 sequences).

---

## 1. Directory Structure & Key Files

* `PROTOCOL.md`: Pre-registered evaluation protocol locked prior to any test-split evaluation (Commit: `d509de0`).
* `folds_summary_table.csv`: Main performance summary across all fixed baselines, single-level policies, and full AFM-RL.
* `primary_comparisons_P1_P5.csv`: Paired statistical tests (P1–P5) with sequence-clustered bootstrap 95% CIs ($B=10,000$) and exact McNemar $p$-values.
* `ablation_table.csv`: Component ablations (L1 preset selection vs. static presets; L2 dynamic budgeting vs. static allocations).
* `level1_selection_stats.csv`: Level 1 policy preset distribution and Shannon entropy per fold and pooled.
* `level2_action_stats.csv`: Level 2 policy step counts, action frequencies, and first-action STOP percentages.
* `pareto_plot_data.csv`: Accuracy vs. computational cost coordinates for all evaluated methods.
* `tables.tex`: Ready-to-compile LaTeX tables corresponding to Tables I–IV.

---

## 2. Step-by-Step Table Reproduction

### Step 1: Deterministic Evaluation Harness
Ensure the OpenCV RANSAC deterministic seeding hook is enabled in `src/metrics.py`:
```bash
python results/phase3/check_headroom_agreement.py
```
This verifies that evaluation matches yield 100.0% reproducible keypoints, success indicators, and corner errors.

### Step 2: Run Frozen Test Evaluation Across Folds 0..4
To run the frozen benchmark across all 5 folds on the held-out test splits (580 pairs total):
```bash
python scripts/run_frozen_benchmark.py
```
*Evaluates `checkpoints_v2/fold{k}` deterministically without ground-truth access during rollouts, outputting `results/phase3/folds_pairs.csv` (25,520 rows).*

### Step 3: Compute Statistical Aggregations & Bootstrap CIs
To regenerate all summary tables, paired bootstrap confidence intervals, and ablation summaries:
```bash
python scripts/aggregate_folds.py --input results/phase3/folds_pairs.csv --out-dir results/phase3
```

### Step 4: Run Dev-Only Headroom & Selector Diagnostics
To evaluate the RANSAC noise floor across 3 independent seeds, measure label-free proxy correlation/AUROC, and benchmark inner-CV learned selectors:
```bash
python results/phase3/dev_headroom_analysis.py
```

### Step 5: Format Publication LaTeX Tables
To compile the CSV outputs into formatted LaTeX tables:
```bash
python results/phase3/generate_latex_tables.py
```
Outputs are written to `results/phase3/tables.tex`.

