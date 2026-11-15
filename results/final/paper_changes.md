# Paper Audit and Revision Change Log

This document records all modifications made to `paper/afm_rl_paper.tex` during Phase 8. Every modification reconciles experimental claims with the frozen, leakage-free empirical records.

---

## 1. Abstract
- **Old Text:**
  > Our audit reveals critical evaluation artifacts in prior implementations, including train-test feature leakage and environment action misalignments. Across a frozen five-fold cross-validation on the 116 sequences of HPatches and zero-shot transfer across 1,500 pairs of MegaDepth, the reinforcement learning policy collapses: four of five folds select a single detector, and budget optimization terminates immediately. Consequently, the learned policy is statistically indistinguishable from fixed ALIKED-500 in accuracy (72.41\% versus 72.59\% Success at 3 pixels on HPatches; 76.05\% versus 76.13\% Success at 10 degrees on MegaDepth) while incurring a 7.9-fold wall-clock overhead. In contrast, a simple label-free geometric cascade achieves 82.13\% Success at 10 degrees on MegaDepth, reducing keypoint consumption by 53.4\% relative to exhaustive extraction and latency by 70.2\% relative to SIFT-4000. Key limitations include single-seed training and evaluation restricted to two MegaDepth scenes.
- **New Text:**
  > Our audit reveals critical evaluation artifacts in earlier implementations, including test-time ground-truth homography leakage and action mapping index permutations in evaluation environments. Across a frozen five-fold cross-validation on the 116 sequences of HPatches and zero-shot transfer across 1,500 pairs of MegaDepth, the reinforcement learning policy collapses: four of five folds select a single detector, and budget optimization terminates prematurely (89.5\% first-action STOP with 1.12 mean steps on HPatches; 80.47\% first-action STOP with 1.25 mean steps on MegaDepth). Consequently, the learned policy is statistically indistinguishable from fixed ALIKED-500 in accuracy (72.41\% versus 72.59\% Success at 3 pixels on HPatches, difference $-0.17\%$ with 95\% CI $[-1.03\%, +0.52\%]$; 76.05\% versus 76.13\% Success at 10 degrees on MegaDepth, difference $-0.08\%$ with 95\% CI $[-0.66\%, +0.50\%]$) while incurring a 7.9-fold median wall-clock latency overhead (2087.5\,ms versus 264.8\,ms on an NVIDIA RTX 3050 Laptop GPU / Intel Core i5-12450H CPU). In contrast, a pre-registered, fold-selected label-free geometric cascade achieves 79.57\% Success at 10 degrees and 58.05 AUC at 10 degrees on MegaDepth with 1628.40 cumulative keypoints (outperforming ALIKED-500 by $+3.44\%$, 95\% CI $[+2.09\%, +4.81\%]$, and SIFT-4000 by $+2.37\%$, 95\% CI $[+0.04\%, +4.70\%]$). An exploratory post-hoc two-stage cascade reaches 82.13\% Success at 10 degrees and 59.94 AUC, reducing keypoint consumption by 53.4\% relative to exhaustive dual extraction (2013.49 versus 4318.61 keypoints) and median latency by 70.2\% relative to SIFT-4000 (433.7\,ms versus 1456.1\,ms). Key limitations include single-seed training, evaluation restricted to two MegaDepth scenes, and omission of Aachen 6-DoF visual localization due to 3D pipeline prerequisites.
- **Evidence:**
  - `results/final/table_hpatches_complete.csv` (Row 11: Full AFM-RL v1, Row 5: ALIKED-500).
  - `results/final/table_megadepth_complete.csv` (Row 7: AFM-RL v1 Rerun, Row 5: ALIKED-500).
  - `results/final/level2_action_stats.csv` (Pooled: 89.48% Step 1 STOP, 1.12 mean steps).
  - `results/final/c2_fold_selected_cascades_summary.csv` (Pooled: 79.57% Succ@10, 58.05 AUC@10, 1628.40 KP).
  - `results/final/fold_selected_cascade_bootstrap.csv` (vs ALIKED-500, SIFT-4000 CIs).
  - `results/final/timing_benchmark.csv` (ALIKED-500: 264.8 ms median, AFM-RL v1: 2087.5 ms median).

---

## 2. Introduction: Audit Items 1–3
- **Old Text:**
  > 1. Train-Test Feature Leakage: Keypoint candidate pools were generated across entire image sequences simultaneously rather than strictly within isolated training splits, leaking target feature distributions into the evaluation pipeline.
  > 2. Environment-Action Misalignment: In zero-shot 3D transfer evaluation, action indices in the evaluation environment were incorrectly mapped to negative increments, artificially forcing keypoint depletion and early stopping.
  > 3. RANSAC Reproducibility Artifacts: Early investigations attributed variance to non-deterministic random number generator seeds; our audit demonstrates that OpenCV's RANSAC uses an internal deterministic seed, and that input correspondence ordering is the actual driver of selection noise ($\pm 0.3\%$ to $\pm 0.8\%$).
- **New Text:**
  > 1. \textbf{Test-Time Ground-Truth Homography Leakage:} In earlier implementation drafts, ground-truth homography accuracy entered the Level 2 observation vector ($\Delta\text{acc}, \Delta U, U_{\text{gap}}$) and the episode scoreboard at test time, resulting in oracle step selection. When evaluated under an honest, label-free proxy protocol, the apparent policy gains completely evaporate.
  > 2. \textbf{Evaluation Action Mapping Permutation Bug:} In zero-shot 3D evaluation on MegaDepth, discrete action indices in \texttt{src/megadepth\_eval\_env.py} were permuted relative to training: action 3 was mapped to +1000 keypoints rather than $\mathrm{STOP}$, observation dimensions 26--33 held mismatched quantities, and $\text{MAX\_STEPS}$ was 5 instead of 15. The policy repeatedly selected action 3 intending to stop, but instead expanded the budget to $\sim$9,650 cumulative keypoints up to step truncation.
  > 3. \textbf{RANSAC Correspondence Ordering Noise:} Run-to-run variations were originally hypothesized to arise from global RNG seed drift via \texttt{cv2.setRNGSeed}. However, rigorous tests demonstrate that OpenCV's \texttt{cv2.findHomography} uses an internal deterministic seed (20 random seeds produce 1 distinct $H$ matrix), whereas random permutations of input correspondence order alter candidate sample draws, inducing selection noise of $\pm 0.3\%$ to $\pm 0.8\%$.
- **Evidence:**
  - `src/level2_env.py` vs `src/megadepth_eval_env.py` parity audit (`results/phase5/a1_env_action_comparison.csv`).
  - `results/phase3b/noise_floor_test.py` output (1 distinct H matrix across 20 calls to `cv2.setRNGSeed`).
  - `results/phase1/verify_splits.py` (verified 81/18/17 sequence disjoint splits, 0 overlap).

---

## 3. Introduction: Aachen Omission Reason & Headline Claims
- **Old Text:**
  > We explicitly note key experimental limitations upfront: the Aachen Day-Night benchmark was omitted due to an irreparably corrupt archive; the RL policies were trained using a single random seed per fold; and 3D pose evaluation was conducted on two representative MegaDepth scenes.
- **New Text:**
  > We explicitly note key experimental limitations upfront: the Aachen Day-Night benchmark was omitted because standard evaluation requires a full 6-DoF visual localization pipeline registered against the 3D model (e.g., via HLoc), which was outside the scope of pairwise matching; the RL policies were trained using a single random seed per fold; and 3D pose evaluation was conducted on two representative MegaDepth scenes.
- **Evidence:**
  - Disk inspection: `data/3D-models/aachen_v_1_1/aachen_v_1_1.nvm` (490 MB) exists and is intact.

---

## 4. Section III-B: Simplified Proxy Removal
- **Old Text:**
  > A simplified variant scales directly with inliers:
  > \begin{equation}
  > p_{\text{geom, simple}} = \min\left(1.0, \frac{n_{\text{inliers}}}{300}\right) \times \frac{n_{\text{inliers}}}{n_{\text{matches}}}.
  > \end{equation}
- **New Text:**
  > (Deleted, keeping only the canonical $p_{\text{geom}}$ formulation in Eq. 11).
- **Evidence:**
  - Instructions Task D.

---

## 5. Section III-C: Protocol v2 Cascade Pre-Registration
- **Old Text:**
  > In our experimental evaluation, candidate cascades $C_1$ through $C_6$ were pre-registered in Protocol~v2:
  > - $C_1$: SIFT-1000 $\to$ SIFT-4000
  > - $C_2$: ALIKED-500 $\to$ ALIKED-1000 $\to$ SIFT-4000
  > - $C_3$: ALIKED-500 $\to$ SIFT-2000
  > - $C_4$: ALIKED-500 $\to$ SIFT-1000 $\to$ ALIKED-2000
  > - $C_5$: SIFT-2000 $\to$ ALIKED-1000
  > - $C_6$: ALIKED-500 $\to$ SIFT-4000
- **New Text:**
  > In our experimental evaluation, candidate cascades $C_1$ through $C_6$ were formally pre-registered in Protocol~v2 (\texttt{results/phase4/PROTOCOL\_v2\_FINAL.md}, commit \texttt{06ca299}, Sat Nov 14 02:00:06 2026):
  > \begin{itemize}
  >     \item $C_1$: ALIKED-500 $\to$ SIFT-2000 $\to$ SIFT-4000
  >     \item $C_2$: ALIKED-500 $\to$ ALIKED-1000 $\to$ SIFT-4000
  >     \item $C_3$: SIFT-1000 $\to$ SIFT-2000 $\to$ SIFT-4000
  >     \item $C_4$: ALIKED-500 $\to$ SIFT-1000 $\to$ ALIKED-2000
  >     \item $C_5$: SIFT-500 $\to$ SIFT-2000 $\to$ SIFT-4000
  >     \item $C_6$: ALIKED-500 $\to$ SIFT-4000
  > \end{itemize}
  > Candidates $C_1$--$C_5$ were systematically evaluated across confidence thresholds $\tau \in \{0.35, 0.40, 0.45, 0.50\}$. Fold-selected cascades were chosen strictly on validation splits under inner cross-validation: validation splits selected configuration $C_4$ at $\tau = 0.40$ on Folds 0, 2, and 3, $C_4$ at $\tau = 0.45$ on Fold 1, and $C_2$ at $\tau = 0.40$ on Fold 4. The two-stage configuration $C_6$ at fixed $\tau = 0.40$ on the entire MegaDepth test set is evaluated and designated explicitly as an exploratory post-hoc investigation.
- **Evidence:**
  - `results/phase4/PROTOCOL_v2_FINAL.md` (Git commit `06ca299`, Section 2).

---

## 6. Section III-D: Hardware and Timing
- **Old Text:**
  > an AMD Ryzen 7 7730U CPU (8 physical cores, 16 threads, 2.0\,GHz base clock) and an NVIDIA GeForce RTX 3050 Laptop GPU (4\,GB GDDR6 VRAM).
- **New Text:**
  > an Intel Core i5-12450H CPU (8 physical cores, 12 threads, 2.0\,GHz base clock) and an NVIDIA GeForce RTX 3050 Laptop GPU (4\,GB GDDR6 VRAM, CUDA 13.3).
- **Evidence:**
  - Hardware audit via PowerShell `Get-CimInstance Win32_Processor` and `nvidia-smi`.

---

## 7. Table 1 & Section IV-A: HPatches Main Table & Results
- **Old Text (Table 1):**
  > Contained unverified baseline values (SIFT-500: 67.59, XFeat-1000: 71.90, Oracle: 80.69).
- **New Text (Table 1):**
  > Embedded `results/final/tables/tab_hpatches_main.tex` with verified values (SIFT-500: 59.66, SIFT-1000: 67.59, SIFT-2000: 72.76, SIFT-4000: 72.93, ALIKED-500: 72.59, ALIKED-1000: 73.62, XFeat-1000: 55.52, Best Fixed VAL: 73.97, SIFT+L2: 70.34, ALIKED+L2: 72.59, AFM-RL v1: 72.41, Exploratory Cascade: 76.21, Oracle Preset: 84.83).
- **Primary Comparisons Updated:**
  - $P_1$ vs Best Fixed VAL: $\Delta\text{Succ}@3\text{px} = -1.55\%$ ($[-5.34\%, +1.90\%]$), $\Delta\text{AUC}@3\text{px} = -2.83$ ($[-5.67\%, -0.29\%]$).
  - $P_2$ vs ALIKED-500: $\Delta\text{Succ}@3\text{px} = -0.17\%$ ($[-1.03\%, +0.52\%]$), $\Delta\text{AUC}@3\text{px} = -0.08$ ($[-0.56\%, +0.38\%]$).
  - $P_3$ vs SIFT-2000: $\Delta\text{Succ}@3\text{px} = -0.34\%$ ($[-4.48\%, +3.45\%]$), $\Delta\text{AUC}@3\text{px} = -2.20$ ($[-5.25\%, +0.74\%]$).
  - $P_4$ vs SIFT-4000: $\Delta\text{Succ}@3\text{px} = -0.52\%$ ($[-4.66\%, +3.28\%]$), $\Delta\text{AUC}@3\text{px} = -4.32$ ($[-7.31\%, -1.54\%]$).
  - $P_5$ vs Oracle Preset: $\Delta\text{Succ}@3\text{px} = -12.41\%$ ($[-16.21\%, -8.97\%]$), $\Delta\text{AUC}@3\text{px} = -13.63$ ($[-16.15, -11.38]$).
- **Added Discussion:**
  - Detailed explanation of why Succ@3px and AUC@3px rank methods differently (Succ@3px is a step threshold at 3.0 px; AUC integrates error up to 3 px, placing high weight on sub-pixel accuracy below 1 px where dense SIFT keypoints excel).
- **Evidence:**
  - `results/final/table_hpatches_complete.csv`.
  - `results/phase4b/b3_bootstrap_comparisons.csv`.

---

## 8. Table 2 & Section IV-B: MegaDepth Main Table & Results
- **Old Text (Table 2):**
  > Contained unverified baseline values (SIFT-500: 35.80, ALIKED-500 AUC5: 39.46, C4 GPU 2-stage).
- **New Text (Table 2):**
  > Embedded `results/final/tables/tab_megadepth_main.tex` with verified values (SIFT-500: 37.87, SIFT-1000: 55.13, SIFT-2000: 66.80, SIFT-4000: 77.20, ALIKED-500: 76.13, ALIKED-1000: 77.67, AFM-RL v1: 76.05, C4 3-stage: 78.87, Fold-Selected Cascade: 79.57, C6 post-hoc: 82.13, Always-Both: 82.53, Oracle-of-Two: 87.53).
- **Headline Fold-Selected Cascade Reported:**
  - Reference-image cluster bootstrap (10,000 resamples):
    - vs ALIKED-500: $+3.44\%$ ($[+2.09\%, +4.81\%]$), $\Delta\text{AUC}@10^\circ = +2.79$ ($[+1.69, +3.90]$), $\Delta\text{KP} = +1130.92$ ($[+1040.77, +1222.04]$).
    - vs ALIKED-1000: $+1.91\%$ ($[+0.04\%, +3.74\%]$), $\Delta\text{AUC}@10^\circ = +2.57$ ($[+1.16, +3.95]$), $\Delta\text{KP} = +651.43$ ($[+560.44, +744.24]$).
    - vs SIFT-4000: $+2.37\%$ ($[+0.04\%, +4.70\%]$), $\Delta\text{AUC}@10^\circ = +3.24$ ($[+1.24, +5.27]$), $\Delta\text{KP} = -2192.74$ ($[-2295.09, -2088.11]$).
- **Evidence:**
  - `results/final/table_megadepth_complete.csv`.
  - `results/final/fold_selected_cascade_bootstrap.csv`.
  - `results/phase5b/a4_bootstrap_comparisons_fixed.csv`.

---

## 9. Table 3 & Section IV-C: Policy Telemetry
- **Old Text:**
  > On HPatches, 100.0\% of episodes terminate at Step 0 with the action $\mathrm{STOP}$, resulting in a mean step count of exactly $1.00$.
- **New Text:**
  > Embedded `results/final/tables/tab_policy_behaviour.tex`.
  > On HPatches, pooled across the 580 test pairs, $89.48\%$ of episodes terminate at Step 1 with the action $\mathrm{STOP}$, yielding a mean step count of $1.12$ (Fold 0: $76.52\%$ STOP, $1.28$ steps; Fold 1: $93.04\%$ STOP, $1.08$ steps; Fold 2: $89.57\%$ STOP, $1.10$ steps; Fold 3: $95.65\%$ STOP, $1.04$ steps; Fold 4: $92.50\%$ STOP, $1.07$ steps). On MegaDepth, $80.47\%$ of episodes terminate at Step 1, with an overall mean step count of $1.25$.
- **Evidence:**
  - `results/phase3/level2_action_stats.csv`.

---

## 10. Table 4 & Section IV-D: Cascade Detailed Results
- **Old Text (Table 4):**
  > Protocol v2 Amendment 1 label, incomplete pooled keypoint count (1608.40 vs 1628.40).
- **New Text (Table 4):**
  > Embedded `results/final/tables/tab_cascade_results.tex` with verified pooled values (Succ@10: 79.57%, AUC@10: 58.05, Cum KP: 1628.40, Stop Rate: 58.67%, Escalation: 41.33%).
- **Evidence:**
  - `results/final/c2_fold_selected_cascades_summary.csv`.

---

## 11. Section IV-E: Computational Cost
- **Old Text:**
  > On an AMD Ryzen 7 7730U CPU, OpenCV SIFT-4000 requires 1456.1\,ms...
- **New Text:**
  > On an Intel Core i5-12450H CPU, OpenCV SIFT-4000 requires a median of 1456.1\,ms (mean 1537.2\,ms, 95th percentile 2262.6\,ms), while SIFT-2000 requires 1286.3\,ms (mean 1320.1\,ms, p95 1980.5\,ms), SIFT-1000 requires 1182.4\,ms (mean 1210.3\,ms, p95 1810.0\,ms), and SIFT-500 requires 948.3\,ms (mean 980.5\,ms, p95 1540.2\,ms). On an NVIDIA RTX 3050 Laptop GPU, ALIKED-500 requires a median of 264.8\,ms (mean 347.4\,ms, p95 896.2\,ms). AFM-RL v1 incurs a median latency of 2087.5\,ms (mean 2105.2\,ms, p95 4370.1\,ms)---a 7.9-fold latency penalty relative to ALIKED-500. Cascade C4 requires median 435.9\,ms (mean 780.4\,ms, p95 2150.0\,ms); Fold-Selected Cascade requires median 645.4\,ms (mean 960.2\,ms, p95 2410.5\,ms); and Cascade C6 requires median 433.7\,ms (mean 1055.5\,ms, p95 3376.6\,ms), reducing median latency by 70.2\% relative to SIFT-4000.
- **Evidence:**
  - `results/final/timing_benchmark.csv`.

---

## 12. Table 5 & Section IV-F: Implementation Pitfalls
- **Old Text:**
  > Title: Methodological Audit and Discrepancy Reconciliation.
  > Contained unverified claims (train/test feature leakage, reward sign flip, cv2.setRNGSeed global RNG drift).
- **New Text:**
  > Title: Pitfalls Found in Our Earlier Implementation.
  > Embedded `results/final/tables/tab_audit_table.tex`.
  > Items 1--5 completely rewritten to state the verified root causes: test-time ground-truth homography leakage, evaluation action mapping permutation bug, OpenCV internal deterministic RANSAC seed with correspondence ordering noise, C4 cascade nomenclature reconciliation, and reference-image cluster bootstrap correction.
- **Evidence:**
  - Parity scripts, test runs, and raw match files in `results/phase5/`, `results/phase5b/`, `results/final/`.

---

## 13. Section V: Limitations & Section VI: Conclusion
- **Old Text:**
  > Referenced corrupt archive for Aachen, Ryzen CPU.
- **New Text:**
  > Updated to reflect: Aachen visual localization 3D model requirement, Intel Core i5-12450H CPU, single-seed policy training, and exact cascade metrics.
