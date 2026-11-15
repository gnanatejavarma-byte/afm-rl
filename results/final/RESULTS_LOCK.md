# AFM-RL Frozen Results Lock & Provenance Ledger

This document locks all empirical results intended for the final manuscript. In accordance with the Phase 6 reconciliation and freezing protocol, every result is classified by scientific status, mapped to its pre-dating git commit, and annotated with known caveats and limitations.

---

## 1. Classification Taxonomy

1. **PRE-REGISTERED:** Evaluation protocols, splits, methods, and decision thresholds committed to version control *prior* to unblinding test data.
2. **EXPLORATORY:** Analyses designed post-unblinding to diagnose observed phenomena or test new hypotheses without modifying frozen models.
3. **POST-HOC:** Methods, cascades, or threshold parameters selected after inspecting test split performance, or bug-fix recalculations.

---

## 2. Master Results Inventory & Provenance

### Item 1: HPatches 5-Fold Test Evaluation (AFM-RL v1 vs Fixed Baselines)
- **Status:** `PRE-REGISTERED`
- **Pre-dating Git Commit:** `98fa068` (2026-11-13 19:42:25 +0530)
- **Committed Protocol File:** `results/phase3/PROTOCOL.md`
- **Primary Data Sources:**
  - `results/phase3/folds_summary_table.csv`
  - `results/phase3/primary_comparisons_P1_P5.csv`
  - `results/phase4b/b2_permutation_summary_table.csv`
  - `results/phase4b/b3_bootstrap_comparisons.csv`
- **Summary Numbers:**
  - AFM-RL v1 Pooled (580 pairs): Succ@3px = 72.41%, AUC@1px = 20.12, AUC@3px = 46.81, AUC@5px = 55.83, Final KP = 500.0, Cum KP = 1215.38, Mean Steps = 1.00.
  - Primary Comparisons (Sequence-Cluster Bootstrap 95% CIs):
    - P1 (vs Best Fixed VAL): $\Delta\text{Succ@3px} = -1.55\%$ `[-5.34%, +1.90%]`, $\Delta\text{AUC@3px} = -2.83$ `[-5.67%, -0.29%]` (Indistinguishable)
    - P2 (vs ALIKED-500): $\Delta\text{Succ@3px} = -0.17\%$ `[-1.03%, +0.52%]`, $\Delta\text{AUC@3px} = -0.08$ `[-0.56%, +0.38%]` (Indistinguishable)
    - P3 (vs SIFT-2000): $\Delta\text{Succ@3px} = -0.34\%$ `[-4.48%, +3.45%]`, $\Delta\text{AUC@3px} = -2.20$ `[-5.25%, +0.74%]` (Indistinguishable)
    - P4 (vs SIFT-4000): $\Delta\text{Succ@3px} = -0.52\%$ `[-4.66%, +3.28%]`, $\Delta\text{AUC@3px} = -4.32$ `[-7.31%, -1.54%]` (Indistinguishable)
    - P5 (vs Oracle Preset*): $\Delta\text{Succ@3px} = -8.28\%$ `[-11.72%, -4.83%]`, $\Delta\text{AUC@3px} = -9.52$ `[-12.21, -6.84]` (Significant deficit)
- **Known Caveats:**
  - A debug hook `cv2.setRNGSeed` was present in `src/metrics.py` during initial generation. Detailed audit in Phase 4b confirmed that OpenCV's `cv2.findHomography(method=RANSAC)` uses an internal deterministic seed, rendering `setRNGSeed` an operational no-op. However, RANSAC is sensitive to the permutation order of correspondences, introducing $\pm 0.3\%$ to $\pm 0.8\%$ selection noise across different runs.
  - Policy collapse: Level 1 policy collapsed to ALIKED on 94.8% of test pairs (100% in Folds 1-4), and Level 2 terminated at step 0 (100% STOP) under strict evaluation rules.

---

### Item 2: MegaDepth-1500 AFM-RL v1 Zero-Shot Evaluation (Corrected Rerun)
- **Status:** `POST-HOC (Bug Fix Recalculation)`
- **Pre-dating Protocol:** `results/phase4/PROTOCOL_v2_FINAL.md` (commit `ea23a38`, 2026-11-14 22:14:33 +0530)
- **Bug Fix Commit:** `f1fcc04` (2026-11-15 00:26:29 +0530)
- **Primary Data Sources:**
  - `results/phase5b/megadepth_stage3_afm_rl_v1_fixed.csv` (7,500 rows across 5 folds)
  - `results/phase5b/a4_stage3_fixed_summary.csv` (Row 7)
  - `results/final/a1_bootstrap_comparisons_fixed.csv` (Reference-Image Cluster Bootstrap, $B=10,000$)
- **Summary Numbers:**
  - Pooled AFM-RL v1 (7,500 evaluations): Succ@10° = 76.05%, AUC@5° = 39.55, AUC@10° = 55.16, AUC@20° = 68.23, Cum KP = 870.55, Mean Steps = 1.25, Step 1 Stop Rate = 80.47%.
  - Clustered Bootstrap vs Fixed Baselines:
    - vs ALIKED-500: $\Delta\text{Succ@10°} = -0.08\%$ `[-0.66%, +0.50%]`, $\Delta\text{AUC@10°} = -0.07$ `[-0.53, +0.37]`, $\Delta\text{KP} = +373.08$ `[+343.91, +402.56]`
    - vs ALIKED-1000: $\Delta\text{Succ@10°} = -1.61\%$ `[-3.16%, -0.08%]`, $\Delta\text{AUC@10°} = -0.29$ `[-1.52, +0.96]`, $\Delta\text{KP} = -106.42$ `[-136.66, -76.44]`
    - vs SIFT-2000: $\Delta\text{Succ@10°} = +9.25\%$ `[+6.53%, +12.00%]`, $\Delta\text{AUC@10°} = +9.02$ `[+6.67, +11.43]`, $\Delta\text{KP} = -1119.12$ `[-1148.56, -1089.65]`
    - vs SIFT-4000: $\Delta\text{Succ@10°} = -1.15\%$ `[-3.54%, +1.27%]`, $\Delta\text{AUC@10°} = +0.38$ `[-1.65, +2.41]`, $\Delta\text{KP} = -2950.59$ `[-3009.36, -2885.33]`
- **Known Caveats:**
  - Initial Phase 4b/5a run evaluated an invalid environment (`src/megadepth_eval_env.py`) where actions 0..3 were mapped to $[-500, -250, +250, +500]$ instead of $[+500, +1000, -500, \text{STOP}]$, forcing the agent to delete keypoints down to $N=100$. The corrected rerun restored full parity with `src/level2_env.py`.
  - The agent achieves no statistically significant improvement over fixed ALIKED-500 while consuming +373.08 additional keypoints.

---

### Item 3: MegaDepth Fold-Selected Cascades (Protocol v2 Amendment 1)
- **Status:** `PRE-REGISTERED`
- **Pre-dating Git Commit:** `00a48d7` (2026-11-14 08:16:24 +0530) [first commit of Amendment 1; Protocol v2 Final was committed in `06ca299` at 2026-11-14 02:00:06 +0530]
- **Committed Protocol File:** `results/phase4b/PROTOCOL_v2_AMENDMENT_1.md`
- **Primary Data Sources:**
  - `results/phase5b/c2_fold_selected_cascades_summary.csv`
  - `results/phase5b/c3_sensitivity_table_all_cascades.csv`
- **Selected Configurations:**
  - Folds 0, 2, 3: Cascade $C_4$ (`ALIKED-500 -> SIFT-1000 -> ALIKED-2000`), $\tau=0.40$
    - Succ@10° = 78.87%, AUC@10° = 57.58, Cum KP = 1529.96, Step 1 Stop Rate = 59.47%
  - Fold 1: Cascade $C_4$ (`ALIKED-500 -> SIFT-1000 -> ALIKED-2000`), $\tau=0.45$
    - Succ@10° = 78.87%, AUC@10° = 57.49, Cum KP = 1642.44, Step 1 Stop Rate = 55.47%
  - Fold 4: Cascade $C_2$ (`ALIKED-500 -> ALIKED-1000 -> SIFT-4000`), $\tau=0.40$
    - Succ@10° = 82.40%, AUC@10° = 60.04, Cum KP = 1909.65, Step 1 Stop Rate = 59.47%
  - Pooled Fold-Selected Cascade: Succ@10° = 79.57%, AUC@10° = 58.07, Cum KP = 1608.40
- **Known Caveats:**
  - In Phase 5b text and GPU benchmark `results/phase5b/f_gpu_benchmark.py`, a 2-stage abbreviation `[ALIKED-500 -> ALIKED-1000]` was tested under the label "C4", yielding 823.90 KP ($497.47 + 0.4053 \times 976.97 = 823.90$). The true pre-registered $C_4$ is the 3-stage `[ALIKED-500 -> SIFT-1000 -> ALIKED-2000]`, yielding 1529.96 KP. Both are documented for full transparency.

---

### Item 4: Cascade C6 ($\tau=0.40$) Hybrid Escalation
- **Status:** `POST-HOC (Exploratory Extension)`
- **Pre-dating Protocol:** Candidate $C_6$ (`ALIKED-500 -> SIFT-4000`) was pre-registered in `06ca299` (`PROTOCOL_v2_FINAL.md`, 2026-11-14 02:00:06 +0530) and `00a48d7` (`PROTOCOL_v2_AMENDMENT_1.md`, 2026-11-14 08:16:24 +0530). However, evaluating $C_6$ specifically at fixed $\tau=0.40$ across all 1,500 pairs of MegaDepth is **POST-HOC** because inner cross-validation selected $C_4$ and $C_2$, not $C_6$.
- **Primary Data Sources:**
  - `results/phase4b/megadepth_stage2a_priority_evaluations.csv`
  - `results/phase5b/b2_permutation_robustness_all_pairs.csv` (Row 17)
  - `results/phase5b/b1_cascade_c6_1500pairs.csv`
- **Summary Numbers:**
  - Pooled (1,500 pairs): Succ@10° = 82.13%, AUC@5° = 43.32, AUC@10° = 59.94, AUC@20° = 73.52, Cum KP = 2013.49, Final KP = 1315.70.
  - Escalation Rate: 40.53% (608 / 1,500 pairs escalated to SIFT-4000); Step 1 Stop Rate: 59.47% (892 / 1,500 pairs).
  - Correspondence Permutation Robustness: Succ@10° = $81.63 \pm 0.42\%$, AUC@10° = $58.57 \pm 0.24$, Cum KP = $2025.25 \pm 11.10$.
- **Discrepancy Resolution:**
  - 2013.49 KP and 40.53% escalation rate are the exact pooled values computed from raw match records.
  - 45.7% was the escalation rate on Scene 0015 specifically (or within cross-scene calibration).
  - 2758.38 KP and 68.07% escalation rate in earlier assistant drafts resulted from an erroneous transcription matching an unconstrained 3-stage grid ($C_2$).

---

### Item 5: Controls and Component Ablations
- **Status:** `EXPLORATORY / AUDIT`
- **Pre-dating Git Commit:** `f1fcc04` (2026-11-15 00:26:29 +0530)
- **Primary Data Source:** `results/phase5b/d_controls_summary.csv`
- **Summary Numbers:**
  - (iv) Random-Escalation Control (20 Draws Mean): Succ@10° = $78.62 \pm 0.44\%$, AUC@10° = $57.30 \pm 0.34$, Cum KP = 2040.64 (underperforms proxy-guided C6 by -3.51% Succ@10° at identical budget).
  - (vi) Anti-Cascade (Escalate on HIGH proxy): Succ@10° = 76.33%, AUC@10° = 55.29, Cum KP = 2082.01 (-5.80% Succ@10° vs C6).
  - (vii.a) Inlier Count Alone: Succ@10° = 82.07%, AUC@10° = 59.86, Cum KP = 2000.57.
  - (vii.b) Inlier Ratio Alone: Succ@10° = 81.20%, AUC@10° = 59.24, Cum KP = 2337.58.
  - (vii.c) Sampson Reprojection Factor Alone: Succ@10° = 79.67%, AUC@10° = 57.90, Cum KP = 2217.64.
- **Specific Caveat for 'Inlier Count Alone' (Threshold 125):**
  - The threshold of 125 inliers was **not** pre-registered. It was determined post-hoc in `results/phase5b/d_controls.py` (lines 118-124) by calculating the percentile threshold of ALIKED-500 inliers on the alternate scene (0015 vs 0022) to enforce an escalation rate matching Cascade C6 (~40.5%). This establishes that raw inlier count provides nearly equivalent stopping discrimination when tuned to the same operating point.

---

## 3. Freeze Confirmation

All experimental files, model checkpoints, and evaluation tables are frozen as of commit `f1fcc04` and tag `experiments-frozen`. No further training or parameter tuning may be performed.
