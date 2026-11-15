import re
from pathlib import Path

TEX_FILE = Path("paper/afm_rl_paper.tex")
CHANGES_MD = Path("results/final/paper_changes_phase9.md")

def apply_edits():
    with open(TEX_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    changes = []

    def record_change(section, old_text, new_text, evidence):
        changes.append({
            "section": section,
            "old_text": old_text.strip(),
            "new_text": new_text.strip(),
            "evidence": evidence.strip()
        })

    # 1. ABSTRACT
    old_abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", content, re.DOTALL)
    if old_abstract_match:
        old_abs = old_abstract_match.group(1).strip()
        new_abs = (
            "Adaptive feature matching aims to dynamically adjust local feature detectors and keypoint budgets "
            "to the difficulty of individual image pairs. We present an empirical audit of adaptive feature matching "
            "via reinforcement learning, a hierarchical framework designed to select feature presets and modulate keypoint budgets. "
            "Our audit reveals evaluation artifacts in preliminary implementations, including test-time ground-truth accuracy leakage "
            "and action mapping permutations in evaluation environments. Under a frozen, label-free evaluation protocol across all "
            "one hundred sixteen sequences of HPatches and one thousand five hundred wide-baseline pairs of MegaDepth, the learned "
            "reinforcement learning policy collapses: cross-validation folds select a single neural detector, and keypoint optimization "
            "terminates early at step one. Consequently, the learned policy yields no detectable accuracy advantage over a fixed "
            "single-detector baseline, achieving seventy-two point four one percent success at three pixels on HPatches and seventy-six "
            "point zero five percent success at ten degrees on MegaDepth, while incurring substantial execution latency overhead. "
            "In contrast, a simple two-stage proxy-guided cascade that attempts a lightweight neural detector before escalating to exhaustive "
            "extraction achieves seventy-nine point five seven percent success on MegaDepth while reducing keypoint consumption. We conclude "
            "that lightweight geometric verification provides a more practical mechanism for adaptive matching than reinforcement learning."
        )
        content = content.replace(old_abs, new_abs)
        record_change("Abstract", old_abs, new_abs, "Phase 9 Prompt: <=200 words, single paragraph, no symbols/brackets/percentages, define abbreviations")

    # 2. CODE AVAILABILITY
    old_code_av = r"\section*{Code Availability}" + "\n" + r"All evaluation protocols, environment parity fixes, pre-registered cascade specifications, and verification test scripts are preserved and available in the project repository."
    new_code_av = r"\section*{Code Availability}" + "\n" + r"All code, evaluation protocols, environment implementations, and verification scripts are available at \url{https://github.com/gnanatejavarma-byte/afm-rl}."
    if old_code_av in content:
        content = content.replace(old_code_av, new_code_av)
        record_change("Code Availability", old_code_av, new_code_av, "Phase 9 prompt: Code availability URL to https://github.com/gnanatejavarma-byte/afm-rl")

    # 3. INTRODUCTION
    # Leak description
    old_intro_leak = r"\item \textbf{Test-Time Ground-Truth Homography Leakage:} In earlier implementation drafts, ground-truth homography accuracy entered the Level 2 observation vector ($\Delta\text{acc}, \Delta U, U_{\text{gap}}$) and the episode scoreboard at test time, resulting in oracle step selection. When evaluated under an honest, label-free proxy protocol, the apparent policy gains completely evaporate."
    new_intro_leak = r"\item \textbf{Test-Time Ground-Truth Homography Leakage:} In a preliminary implementation of our method, ground-truth homography accuracy entered the Level 2 observation vector ($\text{acc}, \Delta\text{acc}, \Delta U, U - U_{\text{best}}$) and the episode scoreboard at test time, resulting in oracle step selection. When evaluated under an honest, label-free proxy protocol, the apparent policy gains evaporate."
    if old_intro_leak in content:
        content = content.replace(old_intro_leak, new_intro_leak)
        record_change("Introduction", old_intro_leak, new_intro_leak, "Phase 8/9 rules: replace earlier drafts with preliminary implementation, exact verified leak variables")

    # Action bug filename removal
    old_intro_action = r"discrete action indices in \texttt{src/megadepth\_eval\_env.py} were permuted"
    new_intro_action = r"discrete action indices in the evaluation environment were permuted"
    if old_intro_action in content:
        content = content.replace(old_intro_action, new_intro_action)
        record_change("Introduction", old_intro_action, new_intro_action, "Phase 9 prompt: delete file names from body")

    # Findings text
    old_find1 = r"statistically indistinguishable from static ALIKED-500"
    new_find1 = r"showing no detectable difference from static ALIKED-500 (the interval excludes differences larger than about one point)"
    if old_find1 in content:
        content = content.replace(old_find1, new_find1)
        record_change("Introduction", old_find1, new_find1, "Phase 9 prompt: replace statistically indistinguishable")

    old_find2 = r"a pre-registered, fold-selected label-free geometric cascade"
    new_find2 = r"a fold-selected label-free geometric cascade (fixed on HPatches before MegaDepth evaluation)"
    if old_find2 in content:
        content = content.replace(old_find2, new_find2)
        record_change("Introduction", old_find2, new_find2, "Phase 9 prompt: replace pre-registered with fixed on HPatches before MegaDepth evaluation")

    old_find3 = r"and reduces median latency by $70.2\%$ relative to SIFT-4000 (433.7\,ms vs.\ 1456.1\,ms)."
    new_find3 = r"and lowers median latency relative to SIFT-4000 (433.7\,ms vs.\ 1456.1\,ms)."
    if old_find3 in content:
        content = content.replace(old_find3, new_find3)
        record_change("Introduction", old_find3, new_find3, "Phase 9 prompt: delete forbidden string 70.2%")

    # Substantial systemic inefficiencies
    content = content.replace("incurs substantial systemic inefficiencies", "incurs notable systemic inefficiencies")

    # 4. RELATED WORK
    old_rw_cascade = r"Early vision systems employed cascade architectures to rapidly reject negative candidates using simple filters before executing complex classifiers."
    new_rw_cascade = r"Early vision systems employed cascade architectures to rapidly reject negative candidates using simple filters before executing complex classifiers~\cite{viola2001rapid}."
    if old_rw_cascade in content:
        content = content.replace(old_rw_cascade, new_rw_cascade)
        record_change("Related Work", old_rw_cascade, new_rw_cascade, "Phase 9 prompt: add citation for Viola-Jones cascade")

    # Add Viola-Jones to bib
    if r"\bibitem{viola2001rapid}" not in content:
        vj_bib = r"""\bibitem{viola2001rapid}
P.~Viola and M.~Jones, ``Rapid object detection using a boosted cascade of simple features,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2001, pp. 511--518.

\bibitem{balntas2017hpatches}"""
        content = content.replace(r"\bibitem{balntas2017hpatches}", vj_bib)
        record_change("Bibliography", "Added viola2001rapid", vj_bib, "Viola-Jones citation")

    # 5. METHODS (Section III)
    # List all 14 presets
    old_presets_desc = r"corresponding to a pre-defined feature configuration preset across classical (SIFT, ORB, BRISK, KAZE, AKAZE) and modern learned extractors (ALIKED, XFeat)."
    new_presets_desc = (
        r"corresponding to one of 14 pre-defined feature presets spanning classical, binary, and learned detectors: "
        r"SIFT, ORB, AKAZE, BRISK, FAST+BRIEF, KAZE, GFTT+BRIEF, GFTT+SIFT, STAR+BRIEF, ORB+FREAK, FAST+FREAK, XFeat, ALIKED, and ORB+BEBLID."
    )
    if old_presets_desc in content:
        content = content.replace(old_presets_desc, new_presets_desc)
        record_change("Methods", old_presets_desc, new_presets_desc, "Phase 9 prompt: list all 14 presets")

    # State Level 1 & 2 trained with proxy-based observations, GT only in reward
    old_l2_obs = r"""The observation vector for Level~2 spans 34 dimensions: the 9 normalized pair statistics $\mathbf{\tilde{s}}$, a 14-dimensional one-hot indicator of the chosen preset, and 11 dynamic state features capturing matching progression:
\begin{align}
\mathbf{o}_t = [&\mathbf{\tilde{s}}, \mathbf{1}_{\text{preset}}, \frac{N_t}{N_{\max}}, \frac{N_{\text{eff}}}{N_{\max}}, \frac{N_{\text{pool}}}{N_{\max}}, \text{acc}_{t-1}, \nonumber \\
&\text{ratio}_{t-1}, \min\left(1.0, \frac{M_{t-1}}{1000}\right), \frac{t}{15}, \Delta\text{acc}, \Delta U, I_{\text{plat}}, U_{\text{gap}}],
\end{align}
where $N_{\text{eff}}$ is the number of keypoints successfully extracted, $N_{\text{pool}}$ is the maximum pool capacity, $\text{ratio}$ is the inlier ratio, $M$ is the number of putative matches, $I_{\text{plat}} = \mathbf{1}(N_{\text{eff}} \ge N_{\text{pool}})$ indicates feature pool exhaustion, and $U_{\text{gap}}$ measures distance from the highest utility observed in the episode."""

    new_l2_obs = r"""Both Level~1 and Level~2 policies were trained strictly with proxy-based observation vectors; ground-truth geometry entered only the scalar reward calculation during training. The observation vector for Level~2 spans 34 dimensions: the 9 normalized pair statistics $\mathbf{\tilde{s}}$, a 14-dimensional one-hot indicator of the chosen preset, and 11 dynamic state features capturing label-free matching progression:
\begin{align}
\mathbf{o}_t = [&\mathbf{\tilde{s}}, \mathbf{1}_{\text{preset}}, \frac{N_t}{N_{\max}}, \frac{N_{\text{eff}}}{N_{\max}}, \frac{N_{\text{pool}}}{N_{\max}}, p_{t-1}, \nonumber \\
&\text{ratio}_{t-1}, \min\left(1.0, \frac{M_{t-1}}{1000}\right), \frac{t}{15}, \Delta p, \Delta U, I_{\text{plat}}, p_{\text{gap}}],
\end{align}
where $N_{\text{eff}}$ is the number of keypoints successfully extracted, $N_{\text{pool}}$ is the maximum pool capacity, $p_{t-1}$ is the previous step's label-free proxy score, $\text{ratio}$ is the RANSAC inlier ratio, $M$ is the putative match count, $\Delta p = p_{t-1} - p_{t-2}$, $I_{\text{plat}} = \mathbf{1}(N_{\text{eff}} \ge N_{\text{pool}})$ indicates feature pool exhaustion, and $p_{\text{gap}} = p_{\text{best}} - p_{t-1}$ measures distance from the highest proxy score observed in the episode."""

    if old_l2_obs in content:
        content = content.replace(old_l2_obs, new_l2_obs)
        record_change("Methods", old_l2_obs, new_l2_obs, "Phase 9 prompt: Level 1 & 2 trained with proxy-based obs, GT only in reward")

    # Matcher description verification against src/ code
    old_matcher_desc = r"All reported metrics evaluate homography estimation under mutual nearest-neighbor matching with Lowe's ratio test (0.75) and RANSAC reprojection threshold 3.0\,pixels."
    new_matcher_desc = (
        r"Feature matching is performed via OpenCV \texttt{BFMatcher} with Lowe's ratio test: floating-point descriptors "
        r"(SIFT, ALIKED, XFeat, KAZE) utilize $L_2$ norm with ratio threshold 0.75, while binary descriptors (ORB, BRISK, BRIEF, FREAK, BEBLID) "
        r"utilize Hamming distance with ratio threshold 0.75, matching the verified implementation in \texttt{src/pipeline\_utils.py} (lines 232--246). "
        r"Homography estimation evaluates RANSAC reprojection threshold 3.0\,pixels."
    )
    if old_matcher_desc in content:
        content = content.replace(old_matcher_desc, new_matcher_desc)
        record_change("Methods", old_matcher_desc, new_matcher_desc, "Phase 9 prompt: matcher description verification against src/ code")

    # Level 1 collapse explanation paragraph via headroom data
    headroom_par = (
        r"The empirical collapse of Level~1 to ALIKED is explained by the distribution of feature headroom and descriptor correlations. "
        r"In our pre-match correlation analysis across HPatches training pairs, the maximum Spearman rank correlation between any pre-match image "
        r"statistic and downstream matching success was only 0.30 (Laplacian blur variance vs. SIFT success). Furthermore, cross-validation "
        r"classifiers (Logistic Regression achieving $72.72\% \pm 1.94\%$ and Gradient Boosting achieving $67.20\% \pm 2.64\%$) failed to surpass "
        r"a static default policy of selecting ALIKED ($73.63\%$). Because ALIKED-500 yields the highest average reward across diverse training sequences, "
        r"the contextual bandit policy converges to selecting ALIKED universally rather than risking suboptimal selections."
    )
    if "The empirical collapse of Level~1 to ALIKED is explained" not in content:
        content = content.replace(
            r"\subsubsection{Level 1 Preset Selection Collapse}",
            r"\subsubsection{Level 1 Preset Selection Collapse}" + "\n" + headroom_par
        )
        record_change("Results: Policy Behaviour", "Added headroom correlation paragraph", headroom_par, "Phase 9 prompt item 7: headroom data, Spearman 0.30, inner-CV selectors")

    # Inlier threshold 125 & cross-scene control provenance
    old_ctrl_125 = r"\item \textbf{Inlier Count Alone (Threshold 125):} Escalating based solely on essential matrix inlier count achieves $82.07\%$ Succ@$10^\circ$ and $59.86$ AUC@$10^\circ$, virtually matching the full proxy ($82.13\%$). In 3D epipolar geometry, raw inlier count provides an exceptionally strong signal of pose convergence because random false matches almost never satisfy 5-point epipolar constraints."
    new_ctrl_125 = (
        r"\item \textbf{Inlier Count Alone (Threshold 125, Cross-Scene):} Escalating based solely on essential matrix inlier count achieves $82.07\%$ Succ@$10^\circ$ "
        r"and $59.86$ AUC@$10^\circ$, closely tracking the full proxy ($82.13\%$). The threshold of 125 inliers was calibrated strictly on Scene~0015 and transferred "
        r"without modification to Scene~0022 as a cross-scene control. In 3D epipolar geometry, raw inlier count provides a strong signal of pose convergence "
        r"because random false matches rarely satisfy 5-point epipolar constraints."
    )
    if old_ctrl_125 in content:
        content = content.replace(old_ctrl_125, new_ctrl_125)
        record_change("Results: Cascade Controls", old_ctrl_125, new_ctrl_125, "Phase 9 prompt: inlier threshold 125 & cross-scene control provenance, remove 'exceptionally'")

    # 6. TABLE I UPDATES
    # SIFT-500/1000 Final KP formatting & ALIKED-1000 AUCs
    old_t1_rows = r"""SIFT-500 & Pooled (580) & 59.66 & 16.40 & 38.89 & 49.86 & 500 & 500 & 1.00 \\
SIFT-1000 & Pooled (580) & 67.59 & 21.09 & 45.42 & 56.65 & 1000 & 1000 & 1.00 \\
SIFT-2000 & Pooled (580) & 72.76 & 24.74 & 49.02 & 60.51 & 1975.23 & 1975.23 & 1.00 \\
SIFT-4000 & Pooled (580) & 72.93 & 26.94 & 51.12 & 62.00 & 3590.86 & 3590.86 & 1.00 \\
ALIKED-500 & Pooled (580) & 72.59 & 21.29 & 46.86 & 59.48 & 497.56 & 497.56 & 1.00 \\
ALIKED-1000 & Pooled (580) & 73.62 & 20.73 & 47.55 & 56.54 & 1000.00 & 1000.00 & 1.00 \\
XFeat-1000 & Pooled (580) & 55.52 & 14.00 & 32.92 & 44.48 & 1000 & 1000 & 1.00 \\
Best Fixed (Fold VAL) & Pooled (580) & 73.97 & 24.66 & 49.62 & 61.04 & 3155.39 & 3155.39 & 1.00 \\
\hline
\multicolumn{9}{l}{\textit{Level 2 Fixed-Preset Ablations}} \\
SIFT + Level 2 & Pooled (580) & 70.34 & 22.81 & 47.17 & 58.93 & 1446.90 & 3803.65 & 2.24 \\
ALIKED + Level 2 & Pooled (580) & 72.59 & 21.35 & 46.78 & 59.58 & 530.67 & 1094.10 & 1.06 \\
\hline
\multicolumn{9}{l}{\textit{Adaptive \& Upper Bound Methods}} \\
Full AFM-RL v1 (Strict STOP) & Pooled (580) & 72.41 & 21.54 & 46.79 & 59.57 & 573.64 & 1215.38 & 1.12 \\
Proxy-Guided Cascade (Exploratory) & Pooled (580) & 76.21 & 21.68 & 49.61 & 58.74 & 680.17 & 920.69 & 1.25 \\
Oracle Preset Selection* & Pooled (580) & 84.83 & 33.50 & 60.41 & 72.04 & 1104.69 & 2731.39 & 1.75 \\
\hline
\multicolumn{9}{l}{\textit{Primary Pre-Registered Comparisons (AFM-RL v1 vs. Baseline, 95\% Clustered Bootstrap CIs)}} \\
P1 vs. Best Fixed VAL & Delta & \multicolumn{2}{l}{-1.55\%\ [-5.34\%, +1.90\%]} & \multicolumn{2}{l}{-2.83\ [-5.67\%, -0.29\%]} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P2 vs. ALIKED-500 & Delta & \multicolumn{2}{l}{-0.17\%\ [-1.03\%, +0.52\%]} & \multicolumn{2}{l}{-0.08\ [-0.56\%, +0.38\%]} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P3 vs. SIFT-2000 & Delta & \multicolumn{2}{l}{-0.34\%\ [-4.48\%, +3.45\%]} & \multicolumn{2}{l}{-2.20\ [-5.25\%, +0.74\%]} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P4 vs. SIFT-4000 & Delta & \multicolumn{2}{l}{-0.52\%\ [-4.66\%, +3.28\%]} & \multicolumn{2}{l}{-4.32\ [-7.31\%, -1.54\%]} & \multicolumn{3}{l}{Succ: Indistinguishable; AUC: Lower} \\
P5 vs. Oracle Preset* & Delta & \multicolumn{2}{l}{-12.41\%\ [-16.21\%, -8.97\%]} & \multicolumn{2}{l}{-13.63\ [-16.15, -11.38]} & \multicolumn{3}{l}{Significantly Lower (CI excludes 0)} \\"""

    new_t1_rows = r"""SIFT-500 & Pooled (580) & 59.66 & 16.40 & 38.89 & 49.86 & 500.00 & 500.00 & 1.00 \\
SIFT-1000 & Pooled (580) & 67.59 & 21.09 & 45.42 & 56.65 & 1000.00 & 1000.00 & 1.00 \\
SIFT-2000 & Pooled (580) & 72.76 & 24.74 & 49.02 & 60.51 & 1975.23 & 1975.23 & 1.00 \\
SIFT-4000 & Pooled (580) & 72.93 & 26.94 & 51.12 & 62.00 & 3590.86 & 3590.86 & 1.00 \\
ALIKED-500 & Pooled (580) & 72.59 & 21.29 & 46.86 & 59.48 & 497.56 & 497.56 & 1.00 \\
ALIKED-1000 & Pooled (580) & 73.62 & 21.11 & 47.55 & 60.36 & 1000.00 & 1000.00 & 1.00 \\
XFeat-1000 & Pooled (580) & 55.52 & 14.00 & 32.92 & 44.48 & 1000.00 & 1000.00 & 1.00 \\
Best Fixed (Fold VAL) & Pooled (580) & 73.97 & 24.66 & 49.62 & 61.04 & 3155.39 & 3155.39 & 1.00 \\
\hline
\multicolumn{9}{l}{\textit{Level 2 Fixed-Preset Ablations}} \\
SIFT + Level 2 & Pooled (580) & 70.34 & 22.81 & 47.17 & 58.93 & 1446.90 & 3803.65 & 2.24 \\
ALIKED + Level 2 & Pooled (580) & 72.59 & 21.35 & 46.78 & 59.58 & 530.67 & 1094.10 & 1.06 \\
\hline
\multicolumn{9}{l}{\textit{Adaptive \& Upper Bound Methods}} \\
Full AFM-RL v1 (Strict STOP) & Pooled (580) & 72.41 & 21.54 & 46.79 & 59.57 & 573.64 & 1215.38 & 1.12 \\
Proxy-Guided Cascade (Seed 0) & Pooled (580) & 76.21 & 22.87 & 49.61 & 62.54 & 680.17 & 920.69 & 1.25 \\
Proxy-Guided Cascade (3-Seed Mean) & Pooled (580) & 75.23 & 21.23 & 47.90 & 61.21 & 543.10 & 622.41 & 1.25 \\
Oracle Preset (N=4000 Ceiling)* & Pooled (580) & 80.69 & 24.97 & 56.33 & 65.31 & 4000.00 & 4000.00 & 1.00 \\
\hline
\multicolumn{9}{l}{\textit{Primary Comparisons (AFM-RL v1 vs. Baseline, 95\% Clustered Bootstrap CIs)}} \\
P1 vs. Best Fixed VAL & Delta & \multicolumn{2}{l}{-1.55\%\ [-5.34\%, +1.90\%]} & \multicolumn{2}{l}{-2.83\ [-5.69\%, -0.29\%]} & \multicolumn{3}{l}{Succ: Indistinguishable; AUC: Lower} \\
P2 vs. ALIKED-500 & Delta & \multicolumn{2}{l}{-0.17\%\ [-1.03\%, +0.52\%]} & \multicolumn{2}{l}{-0.08\ [-0.56\%, +0.38\%]} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P3 vs. SIFT-2000 & Delta & \multicolumn{2}{l}{-0.34\%\ [-4.48\%, +3.45\%]} & \multicolumn{2}{l}{-2.23\ [-5.28\%, +0.73\%]} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P4 vs. SIFT-4000 & Delta & \multicolumn{2}{l}{-0.52\%\ [-4.66\%, +3.28\%]} & \multicolumn{2}{l}{-4.33\ [-7.32\%, -1.56\%]} & \multicolumn{3}{l}{Succ: Indistinguishable; AUC: Lower} \\
P5 vs. Oracle Preset* & Delta & \multicolumn{2}{l}{-8.28\%\ [-11.72\%, -4.83\%]} & \multicolumn{2}{l}{-9.52\ [-12.21, -6.84]} & \multicolumn{3}{l}{Significantly Lower (CI excludes 0)} \\"""

    if old_t1_rows in content:
        content = content.replace(old_t1_rows, new_t1_rows)
        record_change("Results: Table I", old_t1_rows, new_t1_rows, "Table I: consistent KP formatting, ALIKED-1000 updated AUCs, Cascade seed 0 & 3-seed mean, P1 row label per metric, N=4000 oracle")

    # Add footnote to Table I
    t1_cap_old = r"\caption{HPatches 5-fold frozen evaluation across all 116 sequences (580 test pairs). All metrics evaluated under honest label-free protocol without test-time ground truth. Primary comparison deltas (P1--P5) include sequence-clustered bootstrap 95\% confidence intervals (10,000 resamples).}"
    t1_cap_new = r"\caption{HPatches 5-fold frozen evaluation across all 116 sequences (580 test pairs). All metrics evaluated under honest label-free protocol without test-time ground truth. Primary comparison deltas (P1--P5) include sequence-clustered bootstrap 95\% confidence intervals (10,000 resamples). $^*$Label-using upper bound; in-sample.}"
    if t1_cap_old in content:
        content = content.replace(t1_cap_old, t1_cap_new)
        record_change("Results: Table I", t1_cap_old, t1_cap_new, "Table I: explain asterisk in caption")

    # Held-Out-Seed Oracle paragraph in prose
    old_p5_prose = r"\item \textbf{$P_5$ vs.\ Oracle Preset Selection*:} The theoretical upper bound selecting the optimal preset per pair achieves $84.83\%$ Succ@3px and $60.41$ AUC@3px. AFM-RL v1 trails this oracle by a significant margin: $\Delta\text{Succ}@3\text{px} = -12.41\%$ ($[-16.21\%, -8.97\%]$) and $\Delta\text{AUC}@3\text{px} = -13.63$ ($[-16.15, -11.38]$)."
    new_p5_prose = (
        r"\item \textbf{$P_5$ vs.\ Oracle Preset Selection*:} The theoretical upper bound selecting the optimal preset per pair at fixed $N=4000$ pool ceiling achieves $80.69\%$ Succ@3px and $56.33$ AUC@3px. "
        r"AFM-RL v1 trails this upper bound by $\Delta\text{Succ}@3\text{px} = -8.28\%$ ($[-11.72\%, -4.83\%]$) and $\Delta\text{AUC}@3\text{px} = -9.52$ ($[-12.21, -6.84]$). "
        r"When evaluated with a held-out random seed on HPatches to avoid in-sample overfitting, an oracle selector achieves $76.61\%$ Succ@3px at budget $N=1000$ and $78.16\%$ over all configurations, "
        r"compared to $73.62\%$ for static ALIKED-1000."
    )
    if old_p5_prose in content:
        content = content.replace(old_p5_prose, new_p5_prose)
        record_change("Results: Table I Prose", old_p5_prose, new_p5_prose, "Phase 9 prompt item 9: held-out-seed oracle 76.61% and 78.16%, delete captures headroom sentences")

    # Delete headroom sentence in line 269
    content = content.replace("capturing substantial headroom above ALIKED-500 with an average budget of 920.69 cumulative keypoints.", "achieving 76.21% Succ@3px with an average budget of 920.69 cumulative keypoints (and 75.23% Succ@3px with 622.41 keypoints across 3-permutation mean).")

    # Table II formatting & headline rename
    content = content.replace(r"Fold-Selected Cascade (Headline)", r"Fold-Selected Cascade (Mean over 5 Folds)")
    content = content.replace(r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\caption{MegaDepth-1500 benchmark performance", 
                            r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\setlength{\tabcolsep}{3.0pt}" + "\n" + r"\footnotesize" + "\n" + r"\caption{MegaDepth-1500 benchmark performance")

    # Table IV formatting
    content = content.replace(r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\caption{Detailed evaluation of feature cascades on MegaDepth-1500:",
                            r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\setlength{\tabcolsep}{3.0pt}" + "\n" + r"\footnotesize" + "\n" + r"\caption{Detailed evaluation of feature cascades on MegaDepth-1500:")

    # Table III protocol status: replace Pre-registered with Fixed on HPatches before MegaDepth evaluation
    content = content.replace("Fold 0: C4 (\\tau=0.40, AL500 \\to SI1k \\to AL2k) & Pre-registered", "Fold 0: C4 (\\tau=0.40, AL500 \\to SI1k \\to AL2k) & Fixed on HPatches")
    content = content.replace("Fold 1: C4 (\\tau=0.45, AL500 \\to SI1k \\to AL2k) & Pre-registered", "Fold 1: C4 (\\tau=0.45, AL500 \\to SI1k \\to AL2k) & Fixed on HPatches")
    content = content.replace("Fold 2: C4 (\\tau=0.40, AL500 \\to SI1k \\to AL2k) & Pre-registered", "Fold 2: C4 (\\tau=0.40, AL500 \\to SI1k \\to AL2k) & Fixed on HPatches")
    content = content.replace("Fold 3: C4 (\\tau=0.40, AL500 \\to SI1k \\to AL2k) & Pre-registered", "Fold 3: C4 (\\tau=0.40, AL500 \\to SI1k \\to AL2k) & Fixed on HPatches")
    content = content.replace("Fold 4: C2 (\\tau=0.40, AL500 \\to AL1k \\to SI4k) & Pre-registered", "Fold 4: C2 (\\tau=0.40, AL500 \\to AL1k \\to SI4k) & Fixed on HPatches")
    content = content.replace("Pooled Fold-Selected Ensemble & Pre-registered", "Pooled Fold-Selected Ensemble & Fixed on HPatches")

    # Remove forbidden words throughout
    # "severe"
    content = content.replace("MDP exhibits severe premature termination", "MDP exhibits marked premature termination")
    # "substantial"
    content = content.replace("Because ALIKED-500 achieves substantial geometric accuracy", "Because ALIKED-500 achieves high geometric accuracy")
    # Early stopping formulated as explicit hypothesis
    old_hyp = r"This premature stopping stems from the step tax ($c_{\text{step}} = 0.01$) and the keypoint penalty ($\lambda = 0.1$)."
    new_hyp = r"We hypothesize that this premature stopping stems from the step tax ($c_{\text{step}} = 0.01$) and keypoint penalty ($\lambda = 0.1$)."
    content = content.replace(old_hyp, new_hyp)

    # "remarkably" and "genuine"
    content = content.replace("performance is remarkably stable:", "performance is stable across permutations:")
    content = content.replace("This confirms that the cascade's gains are genuine and well outside", "This confirms that the cascade's gains are consistent and outside")

    # Fig 6 caption "significantly shifts"
    content = content.replace("Cascade $C_6$ significantly shifts the trade-off curve", "Cascade $C_6$ shifts the trade-off curve")

    # "70.2%" in text
    content = content.replace(r"reduces median execution latency by \textbf{70.2\%} relative to SIFT-4000", r"reduces median execution latency relative to SIFT-4000")
    content = content.replace(r"and median wall-clock latency by $70.2\%$ relative to brute-force SIFT-4000", r"and median wall-clock latency relative to brute-force SIFT-4000")

    # "operational envelope" and "dominating"
    content = content.replace("Cascade $C_6$ establishes a new operational envelope, dominating both static ALIKED and static SIFT baselines.",
                            "Cascade $C_6$ achieves an improved Pareto trade-off curve compared to both static ALIKED and static SIFT baselines.")

    # 7. SECTION IV-F: RENAME TO "Evaluation Pitfalls", 3 generic points, delete 4 & 5
    old_sec_audit = r"""\subsection{Pitfalls Found in Our Earlier Implementation}
Table~\ref{tab:audit_table} details the reconciliation of discrepancies between earlier drafts and our frozen results:
\begin{enumerate}
    \item \textbf{Test-Time Ground-Truth Leakage (HPatches):} Earlier evaluation scripts computed ground-truth corner error and accuracy against $\mathbf{H}_{\text{gt}}$ inside the evaluation environment, leaking ground truth into dynamic observation features ($\Delta\text{acc}, \Delta U, U_{\text{gap}}$) and selecting the oracle step at test time. Under our frozen protocol without test-time ground truth, AFM-RL v1 achieves $72.41\%$ Succ@3px, which is statistically indistinguishable from fixed ALIKED-500 ($72.59\%$, $95\%$ CI $[-1.03\%, +0.52\%]$).
    \item \textbf{Action Mapping Permutation Bug (MegaDepth):} In \texttt{src/megadepth\_eval\_env.py}, discrete action indices were permuted relative to training: action 3 was mapped to +1000 keypoints instead of $\mathrm{STOP}$, observation dimensions 26--33 held mismatched quantities, and $\text{MAX\_STEPS}$ was set to 5 instead of 15. The policy repeatedly selected action 3 intending to stop, but instead added +1000 keypoints up to truncation. Parity rerun yields $80.47\%$ first-step STOP and $1.25$ mean steps ($870.55$ cumulative keypoints), achieving $76.05\%$ Succ@$10^\circ$.
    \item \textbf{RANSAC Correspondence Ordering Noise:} Global RNG seed changes via \texttt{cv2.setRNGSeed} produce zero variance in \texttt{cv2.findHomography} (20 distinct seeds yield 1 distinct $H$ matrix). Rather, random permutations of correspondence ordering alter candidate sample selection order in OpenCV RANSAC, producing $\pm 0.3\%$ to $\pm 0.8\%$ selection variance.
    \item \textbf{Cascade Nomenclature:} Protocol~v2 Amendment 1 pre-registered a 3-stage $C_4$ (ALIKED-500 $\to$ SIFT-1000 $\to$ ALIKED-2000, 1529.96 KP). An exploratory 2-stage GPU variant was evaluated in early benchmarks (823.90 KP). The 3-stage committed variant is reported canonically.
    \item \textbf{Reference-Image Cluster Bootstrap Estimation:} An early bootstrap implementation averaged pose errors across folds prior to applying the $10^\circ$ threshold, producing an artificial $-1.93\%$ delta against ALIKED-500. Correcting this via reference-image cluster bootstrap over raw per-pair matches confirms the true delta of $-0.08\%$ ($[-0.66\%, +0.50\%]$).
\end{enumerate}"""

    new_sec_audit = r"""\subsection{Evaluation Pitfalls}
Table~\ref{tab:audit_table} details the reconciliation of methodological artifacts identified in a preliminary implementation of our method:
\begin{enumerate}
    \item \textbf{Test-Time Ground-Truth Metric Leakage:} In a preliminary implementation, evaluation scripts computed ground-truth corner error against $\mathbf{H}_{\text{gt}}$ inside the evaluation environment, leaking ground truth into dynamic observation features ($\text{acc}, \Delta\text{acc}, \Delta U, U - U_{\text{best}}$) and selecting the oracle step at test time. Under our frozen protocol without test-time ground truth, AFM-RL v1 achieves $72.41\%$ Succ@3px, which shows no detectable difference from fixed ALIKED-500 ($72.59\%$, $95\%$ CI $[-1.03\%, +0.52\%]$).
    \item \textbf{Action Mapping Permutation Across Environments:} In the zero-shot evaluation environment, discrete action indices were permuted relative to training: action 3 was mapped to +1000 keypoints instead of $\mathrm{STOP}$, observation dimensions 26--33 held mismatched quantities, and $\text{MAX\_STEPS}$ was set to 5 instead of 15. The policy repeatedly selected action 3 intending to stop, but instead added keypoints up to truncation. Parity rerun yields $80.47\%$ first-step STOP and $1.25$ mean steps ($870.55$ cumulative keypoints), achieving $76.05\%$ Succ@$10^\circ$.
    \item \textbf{Non-Determinism and Sampling Variance in Geometric Estimation:} Run-to-run variations in homography estimation were initially suspected to arise from RNG seed drift. Controlled experiments demonstrated that OpenCV's \texttt{cv2.findHomography} uses an internal deterministic seed (20 random seeds yield 1 distinct $H$ matrix). Rather, random permutations of input correspondence ordering alter candidate sample draws in OpenCV RANSAC, producing $\pm 0.3\%$ to $\pm 0.8\%$ selection variance.
\end{enumerate}"""

    if old_sec_audit in content:
        content = content.replace(old_sec_audit, new_sec_audit)
        record_change("Section IV-F", old_sec_audit, new_sec_audit, "Phase 9 prompt: rename to Evaluation Pitfalls, 3 generic points, delete items 4 & 5, replace earlier drafts with preliminary implementation, delete filenames")

    # 8. SECTION V: LIMITATIONS (Exact 8 required points)
    old_limitations = r"""\section{Limitations}
\label{sec:limitations}

We acknowledge several important limitations of this investigation:
\begin{enumerate}
    \item \textbf{Omission of Aachen Day-Night:} The Aachen Day-Night v1.1 benchmark was omitted because standard evaluation requires a full 6-DoF visual localization pipeline registered against the 3D model (e.g., via HLoc), which was outside the scope of pairwise matching.
    \item \textbf{Single Training Seed per Fold:} Reinforcement learning policies were trained with a single random seed per cross-validation fold, precluding analysis of seed-to-seed variance during policy optimization.
    \item \textbf{Domain Mismatch in Training:} RL policies were trained exclusively on planar homographies from HPatches. Zero-shot transfer to MegaDepth evaluated 3D epipolar geometry without domain-specific 3D policy fine-tuning.
    \item \textbf{Limited MegaDepth Scenes:} MegaDepth evaluation was restricted to 1,500 pairs across scenes 0015 and 0022. While these represent diverse urban architecture, wider cross-dataset generalization remains to be explored.
    \item \textbf{RANSAC Correspondence Noise:} The inherent sensitivity of OpenCV RANSAC to match ordering introduces approximately $\pm 0.3\%$ to $\pm 0.8\%$ selection noise, which must be accounted for when interpreting minor performance differences.
    \item \textbf{Feature Metric Clipping:} The 9 image statistics clip Laplacian blur variance at $\pm 5.0$, limiting sensitivity to extreme defocus blur.
    \item \textbf{Confounded Detector Complementarity:} The cascade couples detector architectures (learned ALIKED vs.\ classical SIFT) with differing keypoint budgets (500 vs.\ 4000), not isolating algorithmic complementarity from feature quantity.
    \item \textbf{Platform-Dependent Latency:} Wall-clock execution times reflect specific CPU (Intel Core \texttt{i5-12450H}) and GPU (NVIDIA RTX 3050 Laptop GPU) implementations; edge deployments would exhibit different latency trade-offs.
    \item \textbf{Absence of Learned Matchers:} Due to compute constraints, learned matchers such as LightGlue~\cite{lindenberger2023lightglue} and SuperGlue~\cite{sarlin2020superglue} were not integrated into the cascade.
\end{enumerate}"""

    new_limitations = r"""\section{Limitations}
\label{sec:limitations}

We acknowledge eight specific limitations of this empirical study:
\begin{enumerate}
    \item \textbf{Single Training Seed per Fold:} Reinforcement learning policies were trained with a single random seed per cross-validation fold, precluding empirical analysis of seed-to-seed optimization variance.
    \item \textbf{Restricted MegaDepth Benchmark Scope:} Zero-shot 3D evaluation was restricted to 1,500 pairs sampled across two scenes (0015 and 0022). Wider cross-dataset generalization across diverse outdoor environments remains to be explored.
    \item \textbf{Omission of 6-DoF Localization Benchmarks:} The Aachen Day-Night benchmark was omitted because standard evaluation requires a full multi-view localization pipeline registered against the 3D model (e.g., via HLoc), exceeding the scope of pairwise feature matching.
    \item \textbf{Default Hyperparameter Configurations:} Local feature extractors were evaluated using standard library default hyperparameters without per-dataset fine-tuning.
    \item \textbf{Global Image Statistics State Representation:} The Level~1 contextual bandit state is restricted to 9 global photometric summary statistics, omitting local spatial patch analysis and structural scene parsing.
    \item \textbf{Discretized Escalation Thresholds:} Epipolar proxy escalation thresholds $\tau$ were selected from coarse candidate grids ($\Delta\tau = 0.05$) rather than through continuous threshold optimization.
    \item \textbf{Fixed RANSAC Sample Budgets:} Pose estimation evaluated fixed RANSAC iterations (maximum 1,000 for homography, 0.99999 confidence for essential matrix) without dynamic adaptive sample sizing.
    \item \textbf{Platform-Dependent Latency Profiling:} Wall-clock execution latencies reflect a single workstation hardware configuration (Intel Core \texttt{i5-12450H} CPU and NVIDIA GeForce RTX 3050 Laptop GPU); edge embedded processors would exhibit different trade-off curves.
\end{enumerate}"""

    if old_limitations in content:
        content = content.replace(old_limitations, new_limitations)
        record_change("Limitations", old_limitations, new_limitations, "Phase 9 prompt item 11: 8 required points")

    # Remove script comment lines containing .py
    content = re.sub(r"%.*?\.py.*?\n", "", content)

    # Save updated file
    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {TEX_FILE} with {len(changes)} major revisions.")

    # Write Markdown report
    with open(CHANGES_MD, "w", encoding="utf-8") as f:
        f.write("# Phase 9 Paper Edits and Provenance Report\n\n")
        f.write("| Section | Old Text | New Text | Evidence / Rule |\n")
        f.write("|---|---|---|---|\n")
        for c in changes:
            old_esc = c["old_text"].replace("\n", " ").replace("|", "\\|")[:80] + "..."
            new_esc = c["new_text"].replace("\n", " ").replace("|", "\\|")[:80] + "..."
            evi_esc = c["evidence"].replace("\n", " ").replace("|", "\\|")
            f.write(f"| {c['section']} | {old_esc} | {new_esc} | {evi_esc} |\n")
    print(f"Saved {CHANGES_MD}")

if __name__ == "__main__":
    apply_edits()
