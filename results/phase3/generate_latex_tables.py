import pandas as pd
from pathlib import Path

def generate_latex():
    base_dir = Path("results/phase3")
    
    # 1. Primary Comparisons P1 - P5
    df_p = pd.read_csv(base_dir / "primary_comparisons_P1_P5.csv")
    tex_p = []
    tex_p.append(r"\begin{table*}[t]")
    tex_p.append(r"\centering")
    tex_p.append(r"\caption{\textbf{Primary Statistical Comparisons (P1--P5) on 5-Fold Pooled Test Set (580 Pairs, 116 Sequences).} Differences are computed as $\Delta = \text{Full AFM-RL} - \text{Baseline}$. 95\% confidence intervals are derived from 10,000 sequence-clustered bootstrap resamples; $p$-values are from exact two-sided McNemar tests on paired binary success indicators.}")
    tex_p.append(r"\label{tab:phase3_primary_comparisons}")
    tex_p.append(r"\begin{tabular}{llcccccc}")
    tex_p.append(r"\toprule")
    tex_p.append(r"Label & Baseline Comparison & Full Succ. & Base Succ. & $\Delta$ Succ. [95\% CI] & $\Delta$ AUC@3px [95\% CI] & Final KP (Full / Base) & McNemar $p$ \\")
    tex_p.append(r"\midrule")
    for _, r in df_p.iterrows():
        label = r["Label"].replace("&", r"\&").replace("%", r"\%")
        comp = r["Comparison"].replace("Full AFM-RL (Strict STOP) vs ", "").replace("&", r"\&").replace("%", r"\%")
        f_succ = f"{r['Full_Succ_%']:.2f}\\%"
        b_succ = f"{r['Base_Succ_%']:.2f}\\%"
        d_succ_ci = f"{r['Obs_Succ_Diff_%']:+.2f}\\% {r['Succ_95%_CI']}"
        d_auc_ci = f"{r['Obs_AUC3_Diff_%']:+.2f}\\% {r['AUC3_95%_CI']}"
        kp_ratio = f"{r['Full_Final_KP']:.0f} / {r['Base_Final_KP']:.0f}"
        p_val = f"{r['McNemar_p_val']:.3f}" if r['McNemar_p_val'] >= 0.001 else "$< 0.001$"
        tex_p.append(f"{label} & {comp} & {f_succ} & {b_succ} & {d_succ_ci} & {d_auc_ci} & {kp_ratio} & {p_val} \\\\")
    tex_p.append(r"\bottomrule")
    tex_p.append(r"\end{tabular}")
    tex_p.append(r"\end{table*}")
    
    # 2. Ablation Table
    df_abl = pd.read_csv(base_dir / "ablation_table.csv")
    tex_abl = []
    tex_abl.append(r"\begin{table}[t]")
    tex_abl.append(r"\centering")
    tex_abl.append(r"\caption{\textbf{Component Ablations on 5-Fold Pooled Test Set (580 Pairs).} Evaluating the impact of hierarchical Level~1 preset selection vs.\ fixed preset baselines, and Level~2 dynamic budgeting vs.\ static allocations.}")
    tex_abl.append(r"\label{tab:phase3_ablations}")
    tex_abl.append(r"\begin{tabular}{lcccccc}")
    tex_abl.append(r"\toprule")
    tex_abl.append(r"Method & Succ@3px (\%) & AUC@3px (\%) & Final KP & Cum. KP & Mean Steps \\")
    tex_abl.append(r"\midrule")
    for _, r in df_abl.iterrows():
        m = r["Method"].replace("&", r"\&").replace("%", r"\%")
        s = f"{r['Succ@3px']:.2f} $\\pm$ {r['Succ_Std']:.2f}"
        auc3 = f"{r['AUC@3px']:.2f}"
        fkp = f"{r['Final_KP']:.1f}"
        ckp = f"{r['Cum_KP']:.1f}"
        steps = f"{r['Mean_Steps']:.2f}"
        tex_abl.append(f"{m} & {s} & {auc3} & {fkp} & {ckp} & {steps} \\\\")
    tex_abl.append(r"\bottomrule")
    tex_abl.append(r"\end{tabular}")
    tex_abl.append(r"\end{table}")

    # 3. Behavioral Table
    df_l1 = pd.read_csv(base_dir / "level1_selection_stats.csv")
    df_l2 = pd.read_csv(base_dir / "level2_action_stats.csv")
    tex_beh = []
    tex_beh.append(r"\begin{table}[t]")
    tex_beh.append(r"\centering")
    tex_beh.append(r"\caption{\textbf{Hierarchical Policy Behavioral Audit across Cross-Validation Folds.} Level~1 preset choice distribution and Shannon entropy (bits) alongside Level~2 mean steps and STOP frequency.}")
    tex_beh.append(r"\label{tab:phase3_behavior}")
    tex_beh.append(r"\begin{tabular}{lcccccc}")
    tex_beh.append(r"\toprule")
    tex_beh.append(r"Fold & ALIKED (\%) & SIFT (\%) & Other (\%) & L1 Entropy (bits) & Mean L2 Steps & First-Action STOP (\%) \\")
    tex_beh.append(r"\midrule")
    for _, r in df_l1.iterrows():
        f = r["Fold"]
        al = f"{r['ALIKED_%']:.1f}\\%"
        sift = f"{r['SIFT_%']:.1f}\\%"
        oth = f"{r['Other_%']:.1f}\\%"
        ent = f"{r['Entropy_Bits']:.2f}"
        r2 = df_l2[df_l2["Fold"] == f].iloc[0]
        steps = f"{r2['Mean_Steps']:.2f}"
        fa_stop = f"{r2['FirstAct_STOP_%']:.1f}\\%"
        tex_beh.append(f"{f} & {al} & {sift} & {oth} & {ent} & {steps} & {fa_stop} \\\\")
    tex_beh.append(r"\bottomrule")
    tex_beh.append(r"\end{tabular}")
    tex_beh.append(r"\end{table}")

    out_file = base_dir / "tables.tex"
    with open(out_file, "w") as f:
        f.write("\n\n".join(["\n".join(tex_p), "\n".join(tex_abl), "\n".join(tex_beh)]))
    print(f"Generated LaTeX tables in {out_file}")

if __name__ == "__main__":
    generate_latex()

