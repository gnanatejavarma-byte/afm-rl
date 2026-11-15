import re
from pathlib import Path

TEX_FILE = Path("paper/afm_rl_paper.tex")

def fix_latex():
    with open(TEX_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add \usepackage{url} to preamble
    if r"\usepackage{url}" not in content:
        content = content.replace(r"\usepackage{xcolor}", "\\usepackage{xcolor}\n\\usepackage{url}")
        print("Added \\usepackage{url}")

    # 2. Fix Figure 1 (Algorithm) line 161 overfull hbox
    old_algo_line = r"\STATE $\mathbf{E}_k, \text{mask}_k, \bar{e}_s \leftarrow \text{EssentialRANSAC}(\mathcal{M}_k, \mathbf{K}_1, \mathbf{K}_2, \text{thr}=0.5)$"
    new_algo_line = r"\STATE $\mathbf{E}_k, \mathbf{m}_k, \bar{e}_s \leftarrow \text{EssentialRANSAC}(\mathcal{M}_k, \mathbf{K}_1, \mathbf{K}_2)$"
    if old_algo_line in content:
        content = content.replace(old_algo_line, new_algo_line)
        print("Fixed Algorithm line 161 overfull hbox")

    # 3. Fix Table 1: use \footnotesize and \tabcolsep{2.5pt}
    old_t1_header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3.5pt}
\caption{HPatches 5-fold frozen evaluation across all 116 sequences (580 test pairs). All metrics evaluated under honest label-free protocol without test-time ground truth. Primary comparison deltas (P1--P5) include sequence-clustered bootstrap 95\% confidence intervals (10,000 resamples). $^*$Label-using upper bound; in-sample.}
\label{tab:hpatches_main}
\small
\begin{tabular}{lcccccccc}"""

    new_t1_header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{2.5pt}
\footnotesize
\caption{HPatches 5-fold frozen evaluation across all 116 sequences (580 test pairs). All metrics evaluated under honest label-free protocol without test-time ground truth. Primary comparison deltas (P1--P5) include sequence-clustered bootstrap 95\% confidence intervals (10,000 resamples). $^*$Label-using upper bound; in-sample.}
\label{tab:hpatches_main}
\begin{tabular}{lcccccccc}"""
    if old_t1_header in content:
        content = content.replace(old_t1_header, new_t1_header)
        print("Fixed Table 1 header")

    # 4. Fix Table 2: remove duplicate \small that was overriding \footnotesize, use \tabcolsep{2.5pt}
    old_t2_header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3.0pt}
\footnotesize
\caption{MegaDepth-1500 benchmark performance across 1,500 pairs (Scenes 0015 and 0022). Metrics report pose recall ($\text{Succ}@10^\circ$) and pose AUC curves up to $5^\circ$, $10^\circ$, and $20^\circ$. Computational overhead accounts for cumulative keypoints evaluated and wall-clock execution time on identical dedicated hardware (NVIDIA RTX 3050 Laptop GPU, Intel Core \texttt{i5-12450H} CPU).}
\label{tab:megadepth_main}
\small
\begin{tabular}{lcccccccc}"""

    new_t2_header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{2.5pt}
\footnotesize
\caption{MegaDepth-1500 benchmark performance across 1,500 pairs (Scenes 0015 and 0022). Metrics report pose recall ($\text{Succ}@10^\circ$) and pose AUC curves up to $5^\circ$, $10^\circ$, and $20^\circ$. Computational overhead accounts for cumulative keypoints evaluated and wall-clock execution time on identical dedicated hardware (NVIDIA RTX 3050 Laptop GPU, Intel Core \texttt{i5-12450H} CPU).}
\label{tab:megadepth_main}
\begin{tabular}{lcccccccc}"""
    if old_t2_header in content:
        content = content.replace(old_t2_header, new_t2_header)
        print("Fixed Table 2 header")

    # 5. Fix Table 3: use \footnotesize and \tabcolsep{4.0pt}
    old_t3_header = r"""\begin{table*}[t]
\centering
\caption{Policy behavioral telemetry across HPatches (5 folds) and MegaDepth (zero-shot transfer). Level 1 measures preset selection entropy (in bits) over the 14-preset action space. Level 2 measures initial action distribution and termination step counts.}
\label{tab:policy_behaviour}
\small
\begin{tabular}{lcccccc}"""

    new_t3_header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4.0pt}
\footnotesize
\caption{Policy behavioral telemetry across HPatches (5 folds) and MegaDepth (zero-shot transfer). Level 1 measures preset selection entropy (in bits) over the 14-preset action space. Level 2 measures initial action distribution and termination step counts.}
\label{tab:policy_behaviour}
\begin{tabular}{lcccccc}"""
    if old_t3_header in content:
        content = content.replace(old_t3_header, new_t3_header)
        print("Fixed Table 3 header")

    # 6. Fix Table 4: remove duplicate \small that was overriding \footnotesize, use \tabcolsep{2.5pt}
    old_t4_header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3.0pt}
\footnotesize
\caption{Detailed evaluation of feature cascades on MegaDepth-1500: cross-validation fold selections, reference-image cluster bootstrap comparisons (10,000 resamples), correspondence permutation sensitivity, and ablation controls. All cascades use unsupervised epipolar proxy $\hat{P}_{epi}$ for escalation decisions.}
\label{tab:cascade_results}
\small
\begin{tabular}{lcccccc}"""

    new_t4_header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{2.5pt}
\footnotesize
\caption{Detailed evaluation of feature cascades on MegaDepth-1500: cross-validation fold selections, reference-image cluster bootstrap comparisons (10,000 resamples), correspondence permutation sensitivity, and ablation controls. All cascades use unsupervised epipolar proxy $\hat{P}_{epi}$ for escalation decisions.}
\label{tab:cascade_results}
\begin{tabular}{lcccccc}"""
    if old_t4_header in content:
        content = content.replace(old_t4_header, new_t4_header)
        print("Fixed Table 4 header")

    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully updated {TEX_FILE}")

if __name__ == "__main__":
    fix_latex()
