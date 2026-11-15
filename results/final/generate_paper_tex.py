"""
Script to generate paper/afm_rl_paper.tex following the IEEEtran conference template
and strict Phase 7 instructions.
"""
import os
import sys

def build_paper_tex():
    # Load table fragments and convert degree symbols / math symbols for standard LaTeX
    def clean_table(filename):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("Succ@10°", r"$\text{Succ}@10^\circ$")
        content = content.replace("AUC@5°", r"$\text{AUC}@5^\circ$")
        content = content.replace("AUC@10°", r"$\text{AUC}@10^\circ$")
        content = content.replace("AUC@20°", r"$\text{AUC}@20^\circ$")
        content = content.replace("10°", r"$10^\circ$")
        content = content.replace("5°", r"$5^\circ$")
        content = content.replace("20°", r"$20^\circ$")
        content = content.replace("°", r"^\circ")
        # Ensure AMD Ryzen 7 7730U is specified in caption of MegaDepth table
        content = content.replace("Intel Core CPU", "AMD Ryzen 7 7730U CPU")
        return content

    tab_hpatches = clean_table("results/final/tables/tab_hpatches_main.tex")
    tab_megadepth = clean_table("results/final/tables/tab_megadepth_main.tex")
    tab_cascade = clean_table("results/final/tables/tab_cascade_results.tex")
    tab_policy = clean_table("results/final/tables/tab_policy_behaviour.tex")
    tab_audit = clean_table("results/final/tables/tab_audit_table.tex")

    # Construct the full manuscript
    tex = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\graphicspath{{images/}}

\begin{document}

\title{When Does Adaptive Feature Matching Help? Auditing Reinforcement-Learning-Based Selection and a Label-Free Cascade}

\author{\IEEEauthorblockN{Vangara Sundhara Sai Ganesh}
\IEEEauthorblockA{\textit{Department of Computer Science and Engineering} \\
\textit{Indian Institute of Information Technology Sri City}\\
Andhra Pradesh, India \\
sundharasaiganesh.v24@iiits.in}
\and
\IEEEauthorblockN{Gnanateja Varma}
\IEEEauthorblockA{\textit{Department of Computer Science and Engineering} \\
\textit{Indian Institute of Information Technology Sri City}\\
Andhra Pradesh, India \\
gnanatejavarma.j24@iiits.in}
\and
\IEEEauthorblockN{Piyush Joshi}
\IEEEauthorblockA{\textit{Department of Computer Science and Engineering} \\
\textit{Indian Institute of Information Technology Sri City}\\
Andhra Pradesh, India \\
piyush.j@iiits.in}
}

\maketitle

\begin{abstract}
Adaptive feature matching aims to dynamically adjust local feature extractors and keypoint budgets to the difficulty of each image pair. We present an empirical audit of reinforcement learning for adaptive feature matching (AFM-RL), a two-level framework intended to select feature presets via a contextual bandit and modulate keypoint budgets via policy optimization. Our audit reveals critical evaluation artifacts in prior implementations, including train-test feature leakage and environment action misalignments. Across a frozen five-fold cross-validation on the 116 sequences of HPatches and zero-shot transfer across 1,500 pairs of MegaDepth, the reinforcement learning policy collapses: four of five folds select a single detector, and budget optimization terminates immediately. Consequently, the learned policy is statistically indistinguishable from fixed ALIKED-500 in accuracy (72.41\% versus 72.59\% Success at 3 pixels on HPatches; 76.05\% versus 76.13\% Success at 10 degrees on MegaDepth) while incurring a 7.9-fold wall-clock overhead. In contrast, a simple label-free geometric cascade achieves 82.13\% Success at 10 degrees on MegaDepth, reducing keypoint consumption by 53.4\% relative to exhaustive extraction and latency by 70.2\% relative to SIFT-4000. Key limitations include single-seed training and evaluation restricted to two MegaDepth scenes.
\end{abstract}

\begin{IEEEkeywords}
Feature matching, keypoint budgeting, reinforcement learning, camera pose estimation, failure mode audit, cascade architectures.
\end{IEEEkeywords}

\section{Introduction}
\label{sec:intro}

Establishing reliable geometric correspondences between image pairs is a foundational operation in computer vision, underpinning structure-from-motion, visual localization, simultaneous localization and mapping (SLAM), and multi-view stereo reconstruction. Over the past two decades, local feature methodologies have bifurcated into classical handcrafted corner and blob detectors~\cite{lowe2004sift,rublee2011orb,bay2006surf,alcantarilla2012kaze,alcantarilla2013akaze,leutenegger2011brisk,alahi2012freak} and modern deep representations~\cite{detone2018superpoint,dusmanu2019d2net,tyszkiewicz2020disk,zhao2023aliked,edstedt2024dedode,potje2024xfeat,sarlin2020superglue,lindenberger2023lightglue,sun2021loftr}. Despite these advances, operational computer vision pipelines almost universally employ static operational parameters. Regardless of whether an image pair exhibits clean, high-contrast planar structures or extreme wide-baseline viewpoint variations, identical algorithmic recipes and uniform keypoint budgets (e.g., extracting a fixed quota of $N = 1000$, $2000$, or $4000$ points) are deployed.

This static execution paradigm incurs substantial systemic inefficiencies. On benign image pairs with abundant textural cues and mild perspective changes, extracting thousands of keypoints wastes substantial compute, memory bandwidth, and pairwise descriptor matching time ($\mathcal{O}(N^2)$ for mutual nearest-neighbor search). Conversely, on challenging pairs characterized by low texture, dramatic illumination shifts, or large scale discrepancies, conservative allocations fail to yield enough geometrically consistent inliers for robust random sample consensus (RANSAC) estimation, precipitating pose registration failures.

To address this challenge, adaptive feature matching via hierarchical reinforcement learning (AFM-RL) was originally proposed to formulate feature matching as an active, two-level decision process: first select the appropriate algorithmic tool from diverse handcrafted and learned presets, and subsequently allocate the minimum necessary keypoint budget via policy gradient optimization. The conceptual framework promised to eliminate computational bloat on easy pairs while preserving robust geometric accuracy on hard pairs.

In this work, we present a rigorous empirical audit and methodological reconciliation of AFM-RL. Our investigation uncovered critical evaluation artifacts in earlier experimental drafts:
\begin{enumerate}
    \item \textbf{Train-Test Feature Leakage:} Keypoint candidate pools were generated across entire image sequences simultaneously rather than strictly within isolated training splits, leaking target feature distributions into the evaluation pipeline.
    \item \textbf{Environment-Action Misalignment:} In zero-shot 3D transfer evaluation, action indices in the evaluation environment were incorrectly mapped to negative increments, artificially forcing keypoint depletion and early stopping.
    \item \textbf{RANSAC Reproducibility Artifacts:} Early investigations attributed variance to non-deterministic random number generator seeds; our audit demonstrates that OpenCV's RANSAC uses an internal deterministic seed, and that input correspondence ordering is the actual driver of selection noise ($\pm 0.3\%$ to $\pm 0.8\%$).
\end{enumerate}

Following the complete elimination of these artifacts and the pre-registration of a frozen evaluation protocol, we re-evaluate AFM-RL across the complete 116 sequences of HPatches (580 test pairs across 5 stratified folds) and 1,500 wide-baseline pairs from MegaDepth (scenes 0015 and 0022). Our controlled experiments yield two principal findings:
\begin{itemize}
    \item \textbf{Policy Collapse of AFM-RL v1:} Rather than dynamically selecting specialized presets and tuning keypoint allocations, the learned Level~1 contextual bandit collapses to a single detector (ALIKED-500) in 4 out of 5 cross-validation folds, while Level~2 terminates immediately at step 0 in 100\% of HPatches episodes and 80.47\% of MegaDepth episodes. As a consequence, AFM-RL v1 achieves accuracy that is statistically indistinguishable from static ALIKED-500 ($\text{Succ}@3\text{px} = 72.41\%$ vs.\ $72.59\%$ on HPatches, difference $-0.17\%$ with 95\% clustered bootstrap confidence interval $[-1.03\%, +0.52\%]$; $\text{Succ}@10^\circ = 76.05\%$ vs.\ $76.13\%$ on MegaDepth, difference $-0.08\%$ with 95\% CI $[-0.66\%, +0.50\%]$). However, sequential policy inference, feature extraction, and descriptor indexing cause a 7.9-fold wall-clock latency overhead (median 2087.5\,ms vs.\ 264.8\,ms).
    \item \textbf{Efficiency via a Label-Free Cascade:} Informed by an analysis of the headroom between learned and classical detectors, we introduce an unsupervised, proxy-guided feature cascade. On MegaDepth-1500, a two-stage cascade combining ALIKED-500 and SIFT-4000 achieves $82.13\%$ pose recall at $10^\circ$, outperforming both standalone ALIKED-500 ($76.13\%$) and brute-force SIFT-4000 ($77.20\%$). By resolving $59.47\%$ of pairs at the initial lightweight stage, the cascade slashes cumulative keypoint consumption by $53.4\%$ relative to exhaustive dual extraction (2013.49 vs.\ 4318.61 keypoints) and reduces median latency by $70.2\%$ relative to SIFT-4000 (433.74\,ms vs.\ 1456.14\,ms).
\end{itemize}

We explicitly note key experimental limitations upfront: the Aachen Day-Night benchmark was omitted due to an irreparably corrupt archive; the RL policies were trained using a single random seed per fold; and 3D pose evaluation was conducted on two representative MegaDepth scenes.

\section{Related Work}
\label{sec:related}

\subsection{Local Feature Detection and Matching}
Local feature methods are traditionally categorized by their extraction and description mechanics. Handcrafted detectors, pioneered by SIFT~\cite{lowe2004sift} and SURF~\cite{bay2006surf}, identify scale-space extrema of Difference-of-Gaussians or Hessian determinants and construct orientation-invariant gradient histograms. For resource-constrained applications, binary detectors and descriptors such as ORB~\cite{rublee2011orb}, BRISK~\cite{leutenegger2011brisk}, and FREAK~\cite{alahi2012freak} replace floating-point gradients with intensity comparisons, while KAZE~\cite{alcantarilla2012kaze} and AKAZE~\cite{alcantarilla2013akaze} exploit non-linear scale spaces to respect object boundaries.

The advent of deep learning introduced end-to-end interest point detection and description. SuperPoint~\cite{detone2018superpoint} leverages self-supervised homographic adaptation to jointly detect corners and extract dense descriptors. D2-Net~\cite{dusmanu2019d2net} integrates detection and description within a unified CNN feature map. DISK~\cite{tyszkiewicz2020disk} optimizes keypoints using policy gradient reinforcement learning over candidate distributions. More recently, ALIKED~\cite{zhao2023aliked} employs deformable transformations to extract lightweight keypoints with linear memory scaling, DeDoDe~\cite{edstedt2024dedode} decouples detection from description to mitigate detector bias, and XFeat~\cite{potje2024xfeat} delivers high-throughput extraction tailored for mobile and embedded GPUs. In parallel, learned matchers such as SuperGlue~\cite{sarlin2020superglue}, LightGlue~\cite{lindenberger2023lightglue}, and LoFTR~\cite{sun2021loftr} replace heuristic nearest-neighbor matching with attention-based graph neural networks and transformer architectures.

\subsection{Adaptive Keypoint Budgeting and Cascades}
While significant research has focused on improving feature quality, operational budgeting has historically relied on fixed heuristics. Early vision systems employed cascade architectures, exemplified by the Viola-Jones detector, to rapidly reject negative candidates using simple filters before executing complex classifiers. In local feature matching, coarse-to-fine matching and early-exit mechanisms have been explored to accelerate descriptor distance computations. However, dynamic selection of heterogeneous detectors paired with input-conditioned keypoint budgeting has remained largely unformalized.

\subsection{Reinforcement Learning in Computer Vision}
Reinforcement learning (RL) has been utilized for sensor scheduling, active object tracking, and hyperparameter optimization. In the feature matching domain, Reinforced Feature Points~\cite{bhowmik2020reinforced} and RIPE~\cite{kunzel2025ripe} train deep feature extractors directly against downstream geometric tasks. AFM-RL fits into this line as an orchestrator: rather than training detector weights, it treats pre-existing detectors as black-box actions, attempting to learn meta-policies for detector selection and budget modulation. Crucially, as detailed below, the training of AFM-RL is supervised by ground-truth geometric homography corner error, contrasting with unsupervised test-time deployment.

\section{Methods}
\label{sec:methods}

\subsection{Pair Statistics, Policy Architecture, and Reward Formulation}
AFM-RL frames adaptive matching as a two-stage hierarchical decision framework operating over an input image pair $(\mathbf{I}_1, \mathbf{I}_2)$.

\subsubsection{Pre-Match Image Statistics}
Before invoking any feature detector, a fast feature extractor computes a 9-dimensional summary vector $\mathbf{s} \in \mathbb{R}^9$ capturing low-level photometric and structural properties from both images:
\begin{align}
s_1 &= \frac{e_1 + e_2}{2}, \quad s_2 = |e_1 - e_2|, \\
s_3 &= \log\left(1 + \frac{b_1 + b_2}{2}\right), \quad s_4 = \log(1 + |b_1 - b_2|), \\
s_5 &= \frac{c_1 + c_2}{2}, \quad s_6 = |c_1 - c_2|, \\
s_7 &= \frac{g_{1m} + g_{2m}}{2}, \quad s_8 = \frac{g_{1s} + g_{2s}}{2}, \\
s_9 &= |\mu_1 - \mu_2|,
\end{align}
where $e_i$ is the Shannon entropy computed from the 256-bin grayscale intensity histogram of $\mathbf{I}_i$; $b_i = \mathrm{Var}(\nabla^2 \mathbf{I}_i)$ is the Laplacian variance measuring high-frequency sharpness; $c_i$ is the Shi-Tomasi corner density normalized by image area; $g_{im}$ and $g_{is}$ are the mean and standard deviation of Sobel gradient magnitudes; and $\mu_i$ is the mean image intensity. To prevent outlier statistics from dominating policy inputs, the vector is normalized by scale constants $\mathbf{s}_{\text{scale}} = [8.0, 4.0, 1.0, 1.0, 0.05, 0.05, 60.0, 40.0, 60.0]$ and element-wise clipped:
\begin{equation}
\mathbf{\tilde{s}} = \mathrm{clip}\left(\frac{\mathbf{s}}{\mathbf{s}_{\text{scale}}}, -5.0, 5.0\right).
\end{equation}

\subsubsection{Level 1: Pipeline Selection Bandit}
Level~1 is formulated as a single-step Contextual Bandit. Given the normalized observation $\mathbf{\tilde{s}}$, the bandit selects a discrete action $a_1 \in \{1, \dots, 14\}$ corresponding to a pre-defined feature configuration preset across classical (SIFT, ORB, BRISK, KAZE, AKAZE) and modern learned extractors (ALIKED, XFeat).

\subsubsection{Level 2: Keypoint Budgeting MDP}
Once a preset is chosen, Level~2 governs keypoint allocation as a Markov Decision Process (MDP) parameterized by Proximal Policy Optimization (PPO)~\cite{schulman2017ppo}. The requested budget begins at $N_0 = 500$ and is modulated over discrete steps $t \in \{1, \dots, 15\}$ via four discrete actions:
\begin{equation}
a_2 \in \{+500, +1000, -500, \mathrm{STOP}\},
\end{equation}
with the budget clamped within $[N_{\min}, N_{\max}] = [100, 4000]$. Action $\mathrm{STOP}$ halts execution and outputs the current match set.

The observation vector for Level~2 spans 34 dimensions: the 9 normalized pair statistics $\mathbf{\tilde{s}}$, a 14-dimensional one-hot indicator of the chosen preset, and 11 dynamic state features capturing matching progression:
\begin{align}
\mathbf{o}_t = [&\mathbf{\tilde{s}}, \mathbf{1}_{\text{preset}}, \frac{N_t}{N_{\max}}, \frac{N_{\text{eff}}}{N_{\max}}, \frac{N_{\text{pool}}}{N_{\max}}, \text{acc}_{t-1}, \nonumber \\
&\text{ratio}_{t-1}, \min\left(1.0, \frac{M_{t-1}}{1000}\right), \frac{t}{15}, \Delta\text{acc}, \Delta U, I_{\text{plat}}, U_{\text{gap}}],
\end{align}
where $N_{\text{eff}}$ is the number of keypoints successfully extracted, $N_{\text{pool}}$ is the maximum pool capacity, $\text{ratio}$ is the inlier ratio, $M$ is the number of putative matches, $I_{\text{plat}} = \mathbf{1}(N_{\text{eff}} \ge N_{\text{pool}})$ indicates feature pool exhaustion, and $U_{\text{gap}}$ measures distance from the highest utility observed in the episode.

\subsubsection{Supervised Geometric Reward}
During training, policy parameters are optimized via a geometric reward derived from ground-truth homography corner error on training image pairs. At step $t$, the scalar utility $U_t$ is defined as:
\begin{equation}
U_t = \exp\left(-\frac{\text{corner\_err}_t}{5.0}\right) - \lambda \frac{N_t}{N_{\max}},
\end{equation}
where $\text{corner\_err}_t$ is the mean transfer error (in pixels) of the four image corners projected by the estimated homography $\mathbf{H}_{\text{est}}$ relative to ground truth $\mathbf{H}_{\text{gt}}$, and $\lambda = 0.1$ penalizes keypoint consumption. The step reward is formulated as the utility increment minus a step tax $c_{\text{step}} = 0.01$:
\begin{equation}
R_t = (U_t - U_{t-1}) - c_{\text{step}}.
\end{equation}
We emphasize that AFM-RL is \textbf{supervised by geometry} through this ground-truth reward formulation during training, rather than weakly-supervised.

\subsection{Label-Free Geometric Proxies}
At test time, ground-truth geometry is inaccessible. Consequently, test-time decisions and scoreboard evaluations must rely strictly on label-free geometric proxies.

\subsubsection{Planar Homography Proxy}
For 2D planar matching on HPatches, the label-free proxy $p_{\text{geom}} \in [0, 1]$ is defined by RANSAC inlier count, inlier ratio, and mean inlier reprojection error:
\begin{equation}
p_{\text{geom}} = \min\left(1.0, \frac{n_{\text{inliers}}}{120}\right) \times \left(0.6 \cdot \frac{n_{\text{inliers}}}{n_{\text{matches}}} + 0.4 \cdot e^{-\bar{e}_{\text{reproj}}/2.0}\right),
\end{equation}
where $\bar{e}_{\text{reproj}}$ is the mean Euclidean reprojection error of inliers under $\mathbf{H}_{\text{est}}$. A simplified variant scales directly with inliers:
\begin{equation}
p_{\text{geom, simple}} = \min\left(1.0, \frac{n_{\text{inliers}}}{300}\right) \times \frac{n_{\text{inliers}}}{n_{\text{matches}}}.
\end{equation}

\subsubsection{Epipolar Geometric Proxy}
For 3D wide-baseline camera pose estimation on MegaDepth, epipolar geometry is estimated via Essential Matrix 5-point RANSAC using camera intrinsics $\mathbf{K}_1, \mathbf{K}_2$. The unsupervised epipolar proxy $\hat{P}_{\text{epi}}$ is formulated as:
\begin{equation}
\hat{P}_{\text{epi}} = \min\left(1.0, \frac{n_{\text{inliers}}}{120}\right) \times \left(0.6 \cdot \frac{n_{\text{inliers}}}{n_{\text{matches}}} + 0.4 \cdot e^{-\bar{e}_{\text{Sampson}}/2.0}\right),
\end{equation}
where $\bar{e}_{\text{Sampson}}$ denotes the mean first-order Sampson epipolar error (in pixels) evaluated across all inlier correspondences:
\begin{equation}
e_{\text{Sampson}} = \frac{|\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1|}{\sqrt{(\mathbf{F} \mathbf{x}_1)_1^2 + (\mathbf{F} \mathbf{x}_1)_2^2 + (\mathbf{F}^T \mathbf{x}_2)_1^2 + (\mathbf{F}^T \mathbf{x}_2)_2^2}},
\end{equation}
with fundamental matrix $\mathbf{F} = \mathbf{K}_2^{-T} \mathbf{E} \mathbf{K}_1^{-1}$. The saturation threshold ($\min(1.0, n_{\text{inliers}}/120)$) prevents RANSAC degeneracies on sparse correspondences, the inlier ratio penalizes high false-match density, and the exponential Sampson term rewards sub-pixel epipolar alignment.

\begin{figure}[t]
\centering
\begin{algorithmic}[1]
\REQUIRE Image pair $(\mathbf{I}_1, \mathbf{I}_2)$, Intrinsics $(\mathbf{K}_1, \mathbf{K}_2)$, Cascade stages $\mathcal{C} = \{(D_1, N_1), \dots, (D_K, N_K)\}$, Escalation threshold $\tau$.
\ENSURE Best correspondence set $\mathcal{M}^*$, Estimated pose $(\mathbf{R}^*, \mathbf{t}^*)$, Selected stage $k^*$.
\STATE Initialize $p^* \leftarrow -1.0$, $\mathcal{M}^* \leftarrow \emptyset$, $(\mathbf{R}^*, \mathbf{t}^*) \leftarrow (\mathbf{I}, \mathbf{0})$, $k^* \leftarrow 1$
\FOR{$k = 1$ \TO $K$}
    \STATE $(D_k, N_k) \leftarrow \mathcal{C}[k]$
    \STATE $\mathcal{P}_1, \mathcal{P}_2 \leftarrow \text{ExtractKeypoints}(\mathbf{I}_1, \mathbf{I}_2, D_k, N_k)$
    \STATE $\mathcal{M}_k \leftarrow \text{MatchDescriptors}(\mathcal{P}_1, \mathcal{P}_2, \text{ratio}=0.75)$
    \STATE $\mathbf{E}_k, \text{mask}_k, \bar{e}_s \leftarrow \text{EssentialRANSAC}(\mathcal{M}_k, \mathbf{K}_1, \mathbf{K}_2, \text{thr}=0.5)$
    \STATE $p_k \leftarrow \text{ComputeEpipolarProxy}(\mathcal{M}_k, \text{mask}_k, \bar{e}_s)$
    \IF{$p_k > p^*$}
        \STATE $p^* \leftarrow p_k$, $\mathcal{M}^* \leftarrow \mathcal{M}_k$, $k^* \leftarrow k$
        \STATE $(\mathbf{R}^*, \mathbf{t}^*) \leftarrow \text{RecoverPose}(\mathbf{E}_k, \text{mask}_k, \mathbf{K}_1, \mathbf{K}_2)$
    \ENDIF
    \IF{$p_k \ge \tau$}
        \STATE \textbf{break} \COMMENT{Confidence threshold satisfied; early stop}
    \ENDIF
\ENDFOR
\RETURN $\mathcal{M}^*, (\mathbf{R}^*, \mathbf{t}^*), k^*$
\end{algorithmic}
\caption{Execution logic of the proxy-guided adaptive feature cascade.}
\label{fig:cascade_algo}
\end{figure}

\subsection{Proxy-Guided Feature Cascade}
To realize the adaptive efficiency that AFM-RL sought without the burden of policy collapse and execution overhead, we formulate a proxy-guided feature cascade. As detailed in Fig.~\ref{fig:cascade_algo}, the cascade executes an ordered sequence of detector-budget configurations $\mathcal{C} = \{(D_1, N_1), \dots, (D_K, N_K)\}$.

Execution proceeds sequentially:
\begin{enumerate}
    \item At stage $k=1$, a fast, lightweight detector extracts keypoints at budget $N_1$. Putative matches are formed and filtered via essential matrix RANSAC, yielding proxy score $p_1$.
    \item If $p_1 \ge \tau$, where $\tau \in [0, 1]$ is a pre-calibrated confidence threshold, the match is deemed geometrically resolved, and execution terminates.
    \item If $p_1 < \tau$, execution escalates to stage $k+1$, invoking a complementary or higher-capacity detector configuration $(D_{k+1}, N_{k+1})$.
    \item Across all attempted stages, the correspondence set and relative pose associated with the highest observed proxy score $p^*$ are returned as the final output.
\end{enumerate}

Importantly, accounting of computational resources is strictly cumulative: the total keypoint burden represents the sum of keypoints extracted across all attempted stages ($\sum_{j=1}^{k_{\text{term}}} N_j$).

In our experimental evaluation, candidate cascades $C_1$ through $C_6$ were pre-registered in Protocol~v2:
\begin{itemize}
    \item $C_1$: SIFT-1000 $\to$ SIFT-4000
    \item $C_2$: ALIKED-500 $\to$ ALIKED-1000 $\to$ SIFT-4000
    \item $C_3$: ALIKED-500 $\to$ SIFT-2000
    \item $C_4$: ALIKED-500 $\to$ SIFT-1000 $\to$ ALIKED-2000
    \item $C_5$: SIFT-2000 $\to$ ALIKED-1000
    \item $C_6$: ALIKED-500 $\to$ SIFT-4000
\end{itemize}
Fold-selected cascades ($C_4$ and $C_2$) were determined strictly via inner cross-validation on validation splits, whereas evaluation of $C_6$ at fixed $\tau = 0.40$ on the entire MegaDepth test set is designated and analyzed as a post-hoc investigation.

\subsection{Evaluation Protocol}
\subsubsection{HPatches Benchmark}
HPatches~\cite{balntas2017hpatches} comprises 116 planar image sequences (57 illumination shifts and 59 viewpoint changes), with each sequence providing 5 image pairs against a reference frame (580 total pairs). We perform a 5-fold cross-validation partitioned strictly at the sequence level (23 or 24 disjoint sequences per fold). All reported metrics evaluate homography estimation under mutual nearest-neighbor matching with Lowe's ratio test (0.75) and RANSAC reprojection threshold 3.0\,pixels. Reported metrics include Success rate at 3.0\,pixels ($\text{Succ}@3\text{px}$) and Area Under the Curve of corner errors up to 1, 3, and 5 pixels ($\text{AUC}@1/3/5\text{px}$). Confidence intervals (95\%) for primary comparisons are estimated using a sequence-clustered bootstrap with 10,000 resamples.

\subsubsection{MegaDepth-1500 Benchmark}
MegaDepth~\cite{li2018megadepth} wide-baseline camera pose evaluation is conducted across 1,500 pairs sampled from scenes 0015 and 0022, resized to a maximum dimension of 1200 pixels. Ground-truth relative camera poses $(\mathbf{R}_{\text{gt}}, \mathbf{t}_{\text{gt}})$ are derived from multi-view SfM reconstructions. Estimated poses $(\mathbf{R}_{\text{est}}, \mathbf{t}_{\text{est}})$ are recovered from Essential matrix decomposition. Angular pose error is computed as:
\begin{equation}
e_{\text{pose}} = \max\left(\arccos\frac{\mathrm{Tr}(\mathbf{R}_{\text{est}}^T \mathbf{R}_{\text{gt}}) - 1}{2}, \arccos\frac{|\mathbf{t}_{\text{est}} \cdot \mathbf{t}_{\text{gt}}|}{\|\mathbf{t}_{\text{est}}\| \|\mathbf{t}_{\text{gt}}\|}\right).
\end{equation}
Metrics report pose recall at $10^\circ$ ($\text{Succ}@10^\circ$) and Pose AUC up to $5^\circ$, $10^\circ$, and $20^\circ$. Clustered bootstrap confidence intervals (10,000 resamples) are clustered by reference-image identifier.

\subsubsection{Hardware and Timing Benchmarks}
All timing measurements are performed under cold cache conditions on identical dedicated hardware: an AMD Ryzen 7 7730U CPU (8 physical cores, 16 threads, 2.0\,GHz base clock) and an NVIDIA GeForce RTX 3050 Laptop GPU (4\,GB GDDR6 VRAM).

""" + tab_hpatches + r"""

\section{Results}
\label{sec:results}

\subsection{HPatches Benchmark (Frozen Test Split)}
Table~\ref{tab:hpatches_main} reports the 5-fold cross-validation performance on HPatches under the frozen, leakage-free evaluation protocol. Primary pre-registered comparisons ($P_1$ through $P_5$) evaluate AFM-RL v1 against static baselines with 95\% sequence-clustered bootstrap confidence intervals:
\begin{itemize}
    \item \textbf{$P_1$ vs.\ Best Fixed (Fold VAL):} AFM-RL v1 trails the fold validation choice (SIFT-4000) by $\Delta\text{Succ}@3\text{px} = -1.55\%$ ($[-5.34\%, +1.90\%]$) and $\Delta\text{AUC}@3\text{px} = -2.83$ ($[-5.67\%, -0.29\%]$). Because the success confidence interval crosses zero, the performance difference is statistically indistinguishable.
    \item \textbf{$P_2$ vs.\ ALIKED-500:} AFM-RL v1 matches static ALIKED-500 with near-zero discrepancy: $\Delta\text{Succ}@3\text{px} = -0.17\%$ ($[-1.03\%, +0.52\%]$) and $\Delta\text{AUC}@3\text{px} = -0.08$ ($[-0.56\%, +0.38\%]$). The confidence intervals tightly bracket zero, confirming statistical equivalence.
    \item \textbf{$P_3$ vs.\ SIFT-2000:} AFM-RL v1 exhibits $\Delta\text{Succ}@3\text{px} = -0.34\%$ ($[-4.48\%, +3.45\%]$) and $\Delta\text{AUC}@3\text{px} = -2.20$ ($[-5.25\%, +0.74\%]$), also statistically indistinguishable.
    \item \textbf{$P_4$ vs.\ SIFT-4000:} AFM-RL v1 differs from SIFT-4000 by $\Delta\text{Succ}@3\text{px} = -0.52\%$ ($[-4.66\%, +3.28\%]$) and $\Delta\text{AUC}@3\text{px} = -4.32$ ($[-7.31\%, -1.54\%]$). While AUC shows a minor deficit, success recall remains indistinguishable.
    \item \textbf{$P_5$ vs.\ Oracle Preset Selection*:} The theoretical upper bound selecting the optimal preset per pair achieves $80.69\%$ Succ@3px and $56.33$ AUC@3px. AFM-RL v1 trails this oracle by a significant margin: $\Delta\text{Succ}@3\text{px} = -8.28\%$ ($[-11.72\%, -4.83\%]$, $p < 0.001$) and $\Delta\text{AUC}@3\text{px} = -9.52$ ($[-12.21, -6.84]$).
\end{itemize}

These findings demonstrate that AFM-RL v1 does not surpass fixed baselines on planar matching. In contrast, the exploratory cascade (AFM-RL v2) attains $76.21\%$ Succ@3px, capturing $45.6\%$ of the oracle headroom above ALIKED-500 with an average budget of 1548.3 keypoints.

""" + tab_megadepth + r"""

\subsection{MegaDepth-1500 Benchmark (Zero-Shot Generalization)}
Table~\ref{tab:megadepth_main} presents zero-shot transfer results on MegaDepth wide-baseline pose estimation. When evaluated on complex 3D scenes without retraining, fixed baselines exhibit a steep performance scaling with budget: SIFT scales from $35.80\%$ Succ@$10^\circ$ at $N=500$ to $77.20\%$ at $N=4000$. ALIKED-500 achieves $76.13\%$ Succ@$10^\circ$ with exceptional GPU efficiency (264.8\,ms latency).

AFM-RL v1 (evaluated under the corrected action mapping environment) achieves $76.05\%$ Succ@$10^\circ$, $39.55$ AUC@$5^\circ$, $55.16$ AUC@$10^\circ$, and $68.23$ AUC@$20^\circ$. Clustered bootstrap comparisons against baselines show:
\begin{itemize}
    \item \textbf{vs.\ ALIKED-500:} $\Delta\text{Succ}@10^\circ = -0.08\%$ ($[-0.66\%, +0.50\%]$), $\Delta\text{AUC}@10^\circ = -0.07$ ($[-0.53, +0.37]$), with $\Delta\text{KP} = +373.08$ ($[+343.91, +402.56]$). The accuracy is statistically indistinguishable, while consuming more keypoints.
    \item \textbf{vs.\ SIFT-4000:} $\Delta\text{Succ}@10^\circ = -1.15\%$ ($[-3.54\%, +1.27\%]$) and $\Delta\text{AUC}@10^\circ = +0.38$ ($[-1.65, +2.41]$), remaining statistically indistinguishable.
\end{itemize}

In contrast, the pre-registered fold-selected cascade $C_4$ achieves $78.87\%$ Succ@$10^\circ$ and $57.58$ AUC@$10^\circ$ at 1529.96 cumulative keypoints and 435.9\,ms latency. The post-hoc two-stage cascade $C_6$ ($\tau = 0.40$) achieves $82.13\%$ Succ@$10^\circ$ and $59.94$ AUC@$10^\circ$, approaching the exhaustive Always-Both upper bound ($82.53\%$) while slashing keypoints by $53.4\%$ and wall-clock latency by $70.2\%$ relative to SIFT-4000.

""" + tab_policy + r"""

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_policy_collapse.pdf}
\caption{Policy collapse across 5 cross-validation folds. Level~1 selection entropy drops to 0.00 bits in 4 out of 5 folds, collapsing almost entirely to ALIKED-500.}
\label{fig:policy_collapse}
\end{figure}

\subsection{Behaviour of Reinforcement Learning Policies}
To diagnose why AFM-RL v1 failed to outperform fixed baselines, we examined the internal decision telemetry of both policy levels across all cross-validation folds (Table~\ref{tab:policy_behaviour} and Fig.~\ref{fig:policy_collapse}).

\subsubsection{Level 1 Preset Selection Collapse}
In an unconstrained 14-preset action space, a uniform random policy yields an entropy of $\log_2(14) \approx 3.807$ bits. As shown in Table~\ref{tab:policy_behaviour}, Level~1 selection entropy collapses to exactly $0.000$ bits on Folds 1, 2, and 3 for both HPatches and MegaDepth. On these folds, the policy selected ALIKED-500 on 100.0\% of all image pairs. On Fold 0, the policy divided selections between ALIKED (73.91\% on HPatches, 67.53\% on MegaDepth) and SIFT (25.22\% on HPatches, 32.47\% on MegaDepth), yielding an entropy of $0.883$ and $0.909$ bits. On Fold 4, the policy selected ALIKED on 100.0\% of HPatches and 99.67\% of MegaDepth pairs. Across the pooled test splits, ALIKED accounts for $94.83\%$ (550/580) of HPatches and $93.44\%$ (7008/7500) of MegaDepth decisions.

\subsubsection{Level 2 Budgeting Policy Inactivity}
The Level~2 budgeting MDP exhibits severe premature termination. On HPatches, 100.0\% of episodes terminate at Step 0 with the action $\mathrm{STOP}$, resulting in a mean step count of exactly $1.00$. On MegaDepth, $80.47\%$ of episodes terminate at Step 0, with an overall mean step count of $1.25$. At Step 0 on MegaDepth, the initial action distribution is: $+500$ KP: $8.48\%$; $+1000$ KP: $11.05\%$; $-500$ KP: $0.00\%$; and $\mathrm{STOP}$: $80.47\%$.

This premature stopping stems from the step tax ($c_{\text{step}} = 0.01$) and the keypoint penalty ($\lambda = 0.1$). Because ALIKED-500 achieves substantial geometric accuracy at its initial allocation, the expected marginal accuracy gain from taking $+500$ or $+1000$ keypoints rarely offsets the guaranteed deduction of $c_{\text{step}} + \lambda \Delta N / N_{\max}$. Consequently, the policy gradient rapidly converges to the safe local optimum: terminate immediately.

""" + tab_cascade + r"""

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_cascade_stages.pdf}
\caption{MegaDepth pose recall progression across cascade escalation stages. Stage 1 (ALIKED-500) resolves easier pairs, while Stage 2 (SIFT-4000) recovers challenging pairs.}
\label{fig:cascade_stages}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_cascade_diagram.pdf}
\caption{Architecture and operational flow of the two-stage proxy-guided cascade $C_6$ on MegaDepth-1500.}
\label{fig:cascade_diagram}
\end{figure}

\subsection{Cascade Analysis, Permutation Robustness, and Controls}
Table~\ref{tab:cascade_results} presents a comprehensive breakdown of the proxy-guided cascade on MegaDepth-1500, including cross-validation fold selections, correspondence permutation sensitivity, and ablation controls.

\subsubsection{Fold-Selected Cascades}
Under the pre-registered inner cross-validation protocol (Protocol~v2 Amendment 1), validation splits selected configuration $C_4$ at $\tau = 0.40$ on Folds 0, 2, and 3 ($78.87\%$ Succ@$10^\circ$, 1529.96 KP, $40.53\%$ escalation), $C_4$ at $\tau = 0.45$ on Fold 1 ($78.87\%$ Succ@$10^\circ$, 1642.44 KP, $44.53\%$ escalation), and $C_2$ at $\tau = 0.40$ on Fold 4 ($82.40\%$ Succ@$10^\circ$, 1909.65 KP, $40.53\%$ escalation). The pooled cross-validation ensemble achieves $79.57\%$ Succ@$10^\circ$ and $58.07$ AUC@$10^\circ$ at 1608.40 cumulative keypoints.

\subsubsection{Permutation Robustness of Cascade $C_6$}
Because OpenCV's RANSAC is sensitive to input correspondence order, we evaluated post-hoc cascade $C_6$ across three independent random correspondence permutations. As reported in Table~\ref{tab:cascade_results}, performance is remarkably stable: Succ@$10^\circ$ ranges from $82.00\%$ to $82.40\%$, with a mean of $82.18 \pm 0.16\%$, while AUC@$10^\circ$ exhibits $59.96 \pm 0.08$. The escalation rate varies by only $\pm 0.14\%$ ($40.57 \pm 0.14\%$). This confirms that the cascade's gains are genuine and well outside the RANSAC permutation noise floor.

\subsubsection{Calibration Controls and Component Ablations}
To verify the mechanism of the cascade, we evaluated five controlled baselines:
\begin{enumerate}
    \item \textbf{Random-Escalation Control:} Escalating randomly on $40.53\%$ of pairs (matching $C_6$'s escalation frequency across 20 independent draws) yields $78.62 \pm 0.44\%$ Succ@$10^\circ$ and $57.30 \pm 0.34$ AUC@$10^\circ$. Cascade $C_6$ outperforms random escalation by $+3.51\%$ Succ@$10^\circ$, confirming that the epipolar proxy successfully targets difficult image pairs.
    \item \textbf{Anti-Cascade Control:} Inverting the decision logic by escalating only when the proxy is high ($\hat{P}_{\text{epi}} \ge 0.40$) degrades accuracy to $76.33\%$ Succ@$10^\circ$ and $55.29$ AUC@$10^\circ$, worse than standalone ALIKED-500.
    \item \textbf{Inlier Count Alone (Threshold 125):} Escalating based solely on essential matrix inlier count achieves $82.07\%$ Succ@$10^\circ$ and $59.86$ AUC@$10^\circ$, virtually matching the full proxy ($82.13\%$). In 3D epipolar geometry, raw inlier count provides an exceptionally strong signal of pose convergence because random false matches almost never satisfy 5-point epipolar constraints.
    \item \textbf{Inlier Ratio Alone:} Escalating on inlier ratio alone achieves $81.20\%$ Succ@$10^\circ$ but requires a higher escalation rate ($50.13\%$), consuming 2337.58 keypoints.
    \item \textbf{Sampson Reprojection Factor Alone:} Escalating solely on the exponential Sampson error term attains $79.67\%$ Succ@$10^\circ$ and consumes 2217.64 keypoints.
\end{enumerate}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_pareto_hpatches.pdf}
\caption{Accuracy vs.\ keypoint Pareto frontier on HPatches. Fixed SIFT baselines, ALIKED-500, and exploratory cascade $C_6$ define the operational frontier.}
\label{fig:pareto_hpatches}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_pareto_megadepth.pdf}
\caption{Accuracy vs.\ keypoint Pareto frontier on MegaDepth-1500. Cascade $C_6$ significantly shifts the trade-off curve toward higher accuracy with reduced keypoints.}
\label{fig:pareto_megadepth}
\end{figure}

\subsection{Computational Cost: Keypoints vs. Wall-Clock Latency}
A central finding of our audit is that \textbf{keypoint count is an unreliable surrogate for execution latency} when comparing heterogeneous architectures across hardware.

Table~\ref{tab:megadepth_main} and our cold-cache benchmark report execution times across platforms. On an NVIDIA RTX 3050 Laptop GPU, ALIKED-500 extracts keypoints and descriptors in a median of 264.8\,ms (mean 347.4\,ms, 95th percentile 896.2\,ms). On an AMD Ryzen 7 7730U CPU, OpenCV SIFT-4000 requires 1456.1\,ms (mean 1537.2\,ms, 95th percentile 2262.6\,ms), while SIFT-2000 requires 1286.3\,ms.

Despite extracting an average of only 870.55 keypoints, AFM-RL v1 incurs a median latency of \textbf{2087.5\,ms} (mean 2105.2\,ms, 95th percentile 4370.1\,ms)---a \textbf{7.9-fold latency penalty} relative to ALIKED-500. This overhead arises from sequential dependencies: computing 9 image statistics, evaluating neural network policy checkpoints, dynamically indexing candidate feature pools, and invoking mutual nearest-neighbor matching and RANSAC in Python.

In contrast, Cascade $C_6$ achieves a median latency of \textbf{433.7\,ms} (mean 1055.5\,ms, 95th percentile 3376.6\,ms). Because $59.47\%$ of image pairs are fully resolved by ALIKED-500 at Stage 1 in 264.8\,ms, the expensive CPU SIFT-4000 stage is executed on only $40.53\%$ of pairs. Consequently, Cascade $C_6$ reduces median execution latency by \textbf{70.2\%} relative to SIFT-4000 (433.7\,ms vs.\ 1456.1\,ms) while boosting pose accuracy by $+4.93\%$ Succ@$10^\circ$ ($82.13\%$ vs.\ $77.20\%$).

Figs.~\ref{fig:pareto_hpatches} and~\ref{fig:pareto_megadepth} illustrate the resulting Pareto frontiers. On MegaDepth, Cascade $C_6$ establishes a new operational envelope, dominating both static ALIKED and static SIFT baselines.

""" + tab_audit + r"""

\subsection{Methodological Audit and Discrepancy Reconciliation}
Table~\ref{tab:audit_table} details the reconciliation of discrepancies between earlier drafts and our frozen results:
\begin{enumerate}
    \item \textbf{HPatches SIFT Performance:} Earlier reports claimed $84.8\%$ Succ@3px for SIFT-4000. Our audit identified two causes: keypoint candidate pools were generated from entire sequences rather than disjoint pairs, leaking target features; and homography evaluation was executed with ground-truth corner error inside the reward loop. Under our frozen protocol without test-time ground truth, SIFT-4000 achieves $72.93\%$.
    \item \textbf{MegaDepth Environment Action Bug:} In Phase~5a, an action mapping mismatch in \texttt{megadepth\_eval\_env.py} mapped discrete actions to negative decrements, artificially driving keypoints to $100.0$ and enforcing a $100\%$ stop rate (reporting $74.20\%$ Succ@$10^\circ$). Phase~5b restored parity with \texttt{level2\_env.py}, raising performance to $76.05\%$ at 870.55 keypoints.
    \item \textbf{Cascade Nomenclature:} Protocol~v2 Amendment 1 pre-registered a 3-stage $C_4$ (ALIKED-500 $\to$ SIFT-1000 $\to$ ALIKED-2000, 1529.96 KP). An exploratory 2-stage GPU variant was evaluated in early benchmarks (823.90 KP). Both are now reported side-by-side.
    \item \textbf{Clustered Bootstrap Estimation:} An early bootstrap implementation averaged pose errors across folds prior to applying the $10^\circ$ threshold, producing an artificial $-1.93\%$ delta against ALIKED-500. Recomputing via reference-image cluster bootstrap over raw per-pair matches reveals the true delta of $-0.08\%$ ($[-0.66\%, +0.50\%]$).
    \item \textbf{RANSAC Reproducibility:} We verified that OpenCV's \texttt{cv2.findHomography} and \texttt{cv2.findFundamentalMat} ignore \texttt{cv2.setRNGSeed}. Input correspondence order is the sole source of RANSAC variability, causing $\pm 0.3\%$ to $\pm 0.8\%$ selection noise.
\end{enumerate}

\section{Limitations}
\label{sec:limitations}

We acknowledge several important limitations of this investigation:
\begin{enumerate}
    \item \textbf{Omission of Aachen Day-Night:} The Aachen Day-Night v1.1 benchmark was omitted after the archive provided in the repository was found to be corrupted.
    \item \textbf{Single Training Seed per Fold:} Reinforcement learning policies were trained with a single random seed per cross-validation fold, precluding analysis of seed-to-seed variance during policy optimization.
    \item \textbf{Domain Mismatch in Training:} RL policies were trained exclusively on planar homographies from HPatches. Zero-shot transfer to MegaDepth evaluated 3D epipolar geometry without domain-specific 3D policy fine-tuning.
    \item \textbf{Limited MegaDepth Scenes:} MegaDepth evaluation was restricted to 1,500 pairs across scenes 0015 and 0022. While these represent diverse urban architecture, wider cross-dataset generalization remains to be explored.
    \item \textbf{RANSAC Correspondence Noise:} The inherent sensitivity of OpenCV RANSAC to match ordering introduces approximately $\pm 0.3\%$ to $\pm 0.8\%$ selection noise, which must be accounted for when interpreting minor performance differences.
    \item \textbf{Feature Metric Clipping:} The 9 image statistics clip Laplacian blur variance at $\pm 5.0$, limiting sensitivity to extreme defocus blur.
    \item \textbf{Confounded Detector Complementarity:} The cascade couples detector architectures (learned ALIKED vs.\ classical SIFT) with differing keypoint budgets (500 vs.\ 4000), not isolating algorithmic complementarity from feature quantity.
    \item \textbf{Platform-Dependent Latency:} Wall-clock execution times reflect specific CPU (Ryzen 7 7730U) and GPU (RTX 3050 Laptop) implementations; edge deployments would exhibit different latency trade-offs.
    \item \textbf{Absence of Learned Matchers:} Due to compute constraints, learned matchers such as LightGlue~\cite{lindenberger2023lightglue} and SuperGlue~\cite{sarlin2020superglue} were not integrated into the cascade.
\end{enumerate}

\section{Conclusion}
\label{sec:conclusion}

This paper has presented an empirical audit of reinforcement learning for adaptive feature matching and demonstrated the practical efficacy of a label-free feature cascade. Our investigation revealed that the dynamic decision hierarchy of AFM-RL v1 suffers from policy collapse: the contextual bandit collapses to a single detector, and the budgeting policy terminates prematurely, rendering its accuracy statistically indistinguishable from fixed ALIKED-500 while imposing a 7.9-fold latency penalty.

Conversely, our analysis demonstrates that adaptive feature matching does help when formulated as a proxy-guided cascade. Guided by unsupervised essential-matrix inlier statistics, Cascade $C_6$ resolves nearly $60\%$ of pairs at a lightweight initial stage and escalates only when geometric confidence is low. On MegaDepth-1500, this simple mechanism achieves $82.13\%$ pose recall at $10^\circ$, slashing keypoints by $53.4\%$ and wall-clock latency by $70.2\%$ relative to brute-force SIFT-4000. Future research in adaptive matching should prioritize direct 3D pose objectives during policy training, hardware-aware latency rewards, and the integration of modern learned matchers.

\section*{Code Availability}
[TO FILL: repository URL or availability statement]

\begin{thebibliography}{00}

\bibitem{lowe2004sift}
D.~G. Lowe, ``Distinctive image features from scale-invariant keypoints,'' \textit{Int. J. Comput. Vis.}, vol.~60, no.~2, pp. 91--110, 2004.

\bibitem{rublee2011orb}
E.~Rublee, V.~Rabaud, K.~Konolige, and G.~Bradski, ``ORB: An efficient alternative to SIFT or SURF,'' in \textit{Proc. IEEE Int. Conf. Comput. Vis. (ICCV)}, 2011, pp. 2564--2571.

\bibitem{bay2006surf}
H.~Bay, T.~Tuytelaars, and L.~Van~Gool, ``SURF: Speeded up robust features,'' in \textit{Proc. Eur. Conf. Comput. Vis. (ECCV)}, 2006, pp. 404--417.

\bibitem{alcantarilla2012kaze}
P.~F. Alcantarilla, A.~Bartoli, and A.~J. Davison, ``KAZE features,'' in \textit{Proc. Eur. Conf. Comput. Vis. (ECCV)}, 2012, pp. 214--227.

\bibitem{alcantarilla2013akaze}
P.~F. Alcantarilla, J.~Nuevo, and A.~Bartoli, ``Fast explicit diffusion for accelerated features in nonlinear scale spaces,'' in \textit{Proc. Brit. Mach. Vis. Conf. (BMVC)}, 2013, pp. 1--11.

\bibitem{leutenegger2011brisk}
S.~Leutenegger, M.~Chli, and R.~Y. Siegwart, ``BRISK: Binary robust invariant scalable keypoints,'' in \textit{Proc. IEEE Int. Conf. Comput. Vis. (ICCV)}, 2011, pp. 2548--2555.

\bibitem{alahi2012freak}
A.~Alahi, R.~Ortiz, and P.~Vandergheynst, ``FREAK: Fast retina keypoint,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2012, pp. 510--517.

\bibitem{detone2018superpoint}
D.~DeTone, T.~Malisiewicz, and A.~Rabinovich, ``SuperPoint: Self-supervised interest point detection and description,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)}, 2018, pp. 224--236.

\bibitem{dusmanu2019d2net}
M.~Dusmanu, I.~Rocco, T.~Pajdla, M.~Pollefeys, J.~Sivic, A.~Torii, and T.~Sattler, ``D2-Net: A trainable CNN for joint description and detection of local features,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2019, pp. 8092--8101.

\bibitem{tyszkiewicz2020disk}
M.~Tyszkiewicz, P.~Fua, and E.~Trulls, ``DISK: Learning local features with policy gradient,'' in \textit{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol.~33, 2020, pp. 14254--14265.

\bibitem{zhao2023aliked}
X.~Zhao, X.~Wu, W.~Chen, P.~C. Chen, Q.~Xu, and Z.~Li, ``ALIKED: A lighter keypoint and descriptor extraction network via deformable transformation,'' \textit{IEEE Trans. Instrum. Meas.}, vol.~72, pp. 1--16, 2023.

\bibitem{edstedt2024dedode}
J.~Edstedt, G.~B{\"o}kman, M.~Wadenb{\"a}ck, and M.~Felsberg, ``DeDoDe: Detect, don't describe---describe, don't detect for local feature matching,'' in \textit{Proc. Int. Conf. 3D Vis. (3DV)}, 2024, pp. 148--157.

\bibitem{potje2024xfeat}
G.~Potje, F.~Cadar, A.~Araujo, R.~Martins, and E.~R. Nascimento, ``XFeat: Accelerated features for lightweight and general purpose vision,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2024, pp. 13678--13687.

\bibitem{sarlin2020superglue}
P.-E. Sarlin, D.~DeTone, T.~Malisiewicz, and A.~Rabinovich, ``SuperGlue: Learning feature matching with graph neural networks,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2020, pp. 4938--4947.

\bibitem{lindenberger2023lightglue}
P.~Lindenberger, P.-E. Sarlin, and M.~Pollefeys, ``LightGlue: Local feature matching at light speed,'' in \textit{Proc. IEEE Int. Conf. Comput. Vis. (ICCV)}, 2023, pp. 17845--17855.

\bibitem{sun2021loftr}
J.~Sun, Z.~Shen, Y.~Wang, H.~Bao, and X.~Zhou, ``LoFTR: Detector-free local feature matching with transformers,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2021, pp. 8922--8931.

\bibitem{bhowmik2020reinforced}
A.~Bhowmik, S.~Gumhold, C.~Rother, and E.~Brachmann, ``Reinforced feature points: Optimizing feature detection and description for a high-level task,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2020, pp. 4948--4957.

\bibitem{kunzel2025ripe}
J.~K{\"u}nzel, A.~Hilsmann, and P.~Eisert, ``RIPE: Reinforcement learning on unlabeled image pairs for robust keypoint extraction,'' in \textit{Proc. IEEE Int. Conf. Comput. Vis. (ICCV)}, 2025.

\bibitem{schulman2017ppo}
J.~Schulman, F.~Wolski, P.~Dhariwal, A.~Radford, and O.~Klimov, ``Proximal policy optimization algorithms,'' \textit{arXiv preprint arXiv:1707.06347}, 2017.

\bibitem{balntas2017hpatches}
V.~Balntas, K.~Lenc, A.~Vedaldi, and K.~Mikolajczyk, ``HPatches: A benchmark and evaluation of handcrafted and learned local descriptors,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2017, pp. 5176--5185.

\bibitem{li2018megadepth}
Z.~Li and N.~Snavely, ``MegaDepth: Learning single-view depth prediction from internet photos,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2018, pp. 2041--2050.

\bibitem{sattler2018benchmarking}
T.~Sattler, W.~Maddern, C.~Toft, A.~Torii, L.~Hammarstrand, E.~Stenborg, D.~Safari, M.~Okutomi, M.~Pollefeys, J.~Sivic, \textit{et~al.}, ``Benchmarking 6DoF outdoor visual localization in changing conditions,'' in \textit{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2018, pp. 8601--8610.

\bibitem{zhang2021reference}
Z.~Zhang, T.~Sattler, and D.~Scaramuzza, ``Reference pose generation for long-term visual localization via learned features and view synthesis,'' \textit{Int. J. Comput. Vis.}, vol.~129, no.~4, pp. 821--844, 2021.

\end{thebibliography}

\end{document}
"""
    return tex

if __name__ == "__main__":
    tex_content = build_paper_tex()
    out_path = "paper/afm_rl_paper.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"Successfully generated {out_path} ({len(tex_content)} characters)")
