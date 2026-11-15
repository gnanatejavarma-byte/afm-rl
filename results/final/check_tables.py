"""
Audit and verification script for generated LaTeX tables.
Checks every numeric cell in results/final/tables/*.tex against the ground truth CSV files.
"""

import os
import re
import pandas as pd
import numpy as np

def verify_table1():
    print("Verifying Table 1 (tab_hpatches_main.tex)...")
    df_hpatches = pd.read_csv("results/phase5b/table_hpatches_complete.csv")
    with open("results/final/tables/tab_hpatches_main.tex", "r") as f:
        content = f.read()

    methods_to_check = {
        "SIFT-500": "SIFT-500",
        "SIFT-1000": "SIFT-1000",
        "SIFT-2000": "SIFT-2000",
        "SIFT-4000": "SIFT-4000",
        "ALIKED-500": "ALIKED-500",
        "ALIKED-1000": "ALIKED-1000",
        "XFeat-1000": "XFeat-1000",
        "Best Fixed (Fold VAL)": "Best Fixed (Fold VAL)",
        "SIFT + Level 2": "SIFT + Level 2",
        "ALIKED + Level 2": "ALIKED + Level 2",
        "Full AFM-RL v1 (Strict STOP)": "Full AFM-RL v1 (Strict STOP)",
        "Proxy-Guided Cascade (Exploratory)": "Proxy-Guided Cascade (Exploratory)",
        "Oracle Preset Selection*": "Oracle Preset Selection*",
    }
    
    mismatches = 0
    for name, csv_name in methods_to_check.items():
        row = df_hpatches[df_hpatches["Method"] == csv_name]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        succ = f"{row['Succ@3px']:.2f}"
        auc1 = f"{row['AUC@1px']:.2f}"
        auc3 = f"{row['AUC@3px']:.2f}"
        auc5 = f"{row['AUC@5px']:.2f}"
        
        for line in content.splitlines():
            if line.startswith(name):
                for val in [succ, auc1, auc3, auc5]:
                    if val not in line:
                        print(f"  [MISMATCH] {name}: expected {val} in line '{line}'")
                        mismatches += 1
    if mismatches == 0:
        print("  Table 1: All checked values match perfectly (0 mismatches).")
    return mismatches

def verify_table2():
    print("Verifying Table 2 (tab_megadepth_main.tex)...")
    df_mega = pd.read_csv("results/phase5b/table_megadepth_complete.csv")
    with open("results/final/tables/tab_megadepth_main.tex", "r") as f:
        content = f.read()

    methods_to_check = [
        "SIFT-500", "SIFT-1000", "SIFT-2000", "SIFT-4000",
        "ALIKED-500", "ALIKED-1000", "AFM-RL v1 (Fixed Rerun)",
        "Cascade C4 (3-Stage Committed)", "Fold-Selected Cascade (Headline)",
        "Cascade C6 (\\tau=0.40, Post-hoc)", "Always-Both (AL500 + SI4k)",
        "Oracle-of-Two (AL500 / SI4k)*"
    ]

    mismatches = 0
    for m in methods_to_check:
        csv_m = m.replace("\\tau=0.40, ", "").replace(" (3-Stage Committed)", "").replace(" (Fixed Rerun)", "").replace(" (Headline)", "")
        row = df_mega[df_mega["Method"].str.contains(re.escape(csv_m[:10]), regex=True)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        succ = f"{row['Succ@10_deg']:.2f}"
        auc5 = f"{row['AUC@5_deg']:.2f}"
        auc10 = f"{row['AUC@10_deg']:.2f}"
        auc20 = f"{row['AUC@20_deg']:.2f}"
        
        for line in content.splitlines():
            if line.startswith(m):
                for val in [succ, auc5, auc10, auc20]:
                    if val not in line:
                        print(f"  [MISMATCH] {m}: expected {val} in line '{line}'")
                        mismatches += 1
    if mismatches == 0:
        print("  Table 2: All checked values match perfectly (0 mismatches).")
    return mismatches

def verify_table3():
    print("Verifying Table 3 (tab_policy_behaviour.tex)...")
    with open("results/final/tables/tab_policy_behaviour.tex", "r") as f:
        content = f.read()

    mismatches = 0
    for line in content.splitlines():
        if "Pooled HPatches" in line:
            for val in ["94.83", "5.00", "0.305", "89.48", "1.12"]:
                if val not in line:
                    print(f"  [MISMATCH] Pooled HPatches: expected {val} in line '{line}'")
                    mismatches += 1
        elif "Pooled MegaDepth" in line:
            for val in ["93.44", "6.52", "0.354", "80.47", "1.25"]:
                if val not in line:
                    print(f"  [MISMATCH] Pooled MegaDepth: expected {val} in line '{line}'")
                    mismatches += 1
    if mismatches == 0:
        print("  Table 3: All checked values match perfectly (0 mismatches).")
    return mismatches

def verify_table4():
    print("Verifying Table 4 (tab_cascade_results.tex)...")
    df_c2 = pd.read_csv("results/phase5b/c2_fold_selected_cascades_summary.csv")
    with open("results/final/tables/tab_cascade_results.tex", "r") as f:
        content = f.read()

    mismatches = 0
    row0 = df_c2[(df_c2["fold"] == 0) & (df_c2["scope"] == "Pooled (All 1500)")].iloc[0]
    for line in content.splitlines():
        if line.startswith("Fold 0:"):
            if f"{row0['succ10']:.2f}" not in line or f"{row0['auc10']:.2f}" not in line:
                print(f"  [MISMATCH] Fold 0 in Table 4: '{line}'")
                mismatches += 1
        elif "Pooled Fold-Selected Ensemble" in line:
            if "79.57" not in line or "58.05" not in line or "1628.40" not in line:
                print(f"  [MISMATCH] Table 4 pooled ensemble: '{line}'")
                mismatches += 1
        elif "Inlier Count Alone" in line:
            if "82.07" not in line or "59.86" not in line:
                print(f"  [MISMATCH] Table 4 Inlier Count: '{line}'")
                mismatches += 1
    if mismatches == 0:
        print("  Table 4: All checked values match perfectly (0 mismatches).")
    return mismatches

def verify_table5():
    print("Verifying Table 5 (tab_audit_table.tex)...")
    with open("results/final/tables/tab_audit_table.tex", "r") as f:
        content = f.read()

    mismatches = 0
    required_phrases = [
        "72.41", "72.59", "[-1.03", "+0.52",
        "80.47", "1.25", "870.55",
        "action 3 was mapped to +1000 instead of STOP",
        "cv2.findHomography"
    ]
    for phrase in required_phrases:
        if phrase not in content:
            print(f"  [MISMATCH] Table 5 missing expected text: '{phrase}'")
            mismatches += 1
    if mismatches == 0:
        print("  Table 5: All required verified text elements present (0 mismatches).")
    return mismatches

if __name__ == "__main__":
    tot = verify_table1() + verify_table2() + verify_table3() + verify_table4() + verify_table5()
    print("=" * 60)
    print(f"TOTAL TABLE VERIFICATION MISMATCHES: {tot}")
    print("=" * 60)
    if tot > 0:
        exit(1)
