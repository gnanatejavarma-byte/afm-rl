import subprocess
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_frozen_benchmark():
    print("=" * 80)
    print("PHASE 3: FROZEN TEST EVALUATION ON FOLDS 0..4 (580 TEST PAIRS)")
    print("=" * 80)

    out_csv = ROOT / "results" / "phase3" / "folds_pairs.csv"
    if out_csv.exists():
        print(f"Notice: {out_csv} already exists. Folds that are already present will not be overwritten.")

    python_exe = sys.executable

    t_start = time.time()
    for fold in range(5):
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Starting Benchmark for Fold {fold} on TEST split...")
        cmd = [
            python_exe, "-u", "scripts/benchmark_folds.py",
            "--fold", str(fold),
            "--split", "test",
            "--out-csv", str(out_csv),
        ]
        t0 = time.time()
        ret = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.time() - t0
        if ret.returncode != 0:
            print(f"Error running fold {fold}! Exit code: {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"[{time.strftime('%H:%M:%S')}] Fold {fold} completed in {elapsed:.1f}s ({elapsed/60:.2f}m)")

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL 5 FOLDS BENCHMARKED IN {total_elapsed:.1f}s ({total_elapsed/60:.2f}m)!")
    print(f"Output saved to: {out_csv}")
    print("=" * 80)

    print("\nRunning aggregate_folds.py on results/phase3/folds_pairs.csv...")
    cmd_agg = [
        python_exe, "-u", "scripts/aggregate_folds.py",
        "--csv", str(out_csv),
        "--out-dir", str(ROOT / "results" / "phase3"),
    ]
    subprocess.run(cmd_agg, cwd=str(ROOT))

if __name__ == "__main__":
    run_frozen_benchmark()

