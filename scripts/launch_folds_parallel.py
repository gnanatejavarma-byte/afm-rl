import subprocess
import os
import sys
import time
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent

def launch_folds(max_parallel=2, threads_per_proc=4):
    print("=" * 70)
    print("AFM-RL: 5-FOLD PARALLEL TRAINING QUEUE MANAGER")
    print(f"  Folds to run:    [0, 1, 2, 3, 4]")
    print(f"  Max Concurrency: {max_parallel}")
    print(f"  Threads/proc:    {threads_per_proc}")
    print("=" * 70)

    folds_queue = [0, 1, 2, 3, 4]
    active_processes = {} # fold -> subprocess.Popen

    log_dir = ROOT / "results" / "phase2c"
    log_dir.mkdir(parents=True, exist_ok=True)
    manager_log = open(log_dir / "folds_manager.log", "w", encoding="utf-8")

    def log_print(msg):
        print(msg)
        sys.stdout.flush()
        manager_log.write(msg + "\n")
        manager_log.flush()

    # Function to start a fold process
    def start_fold(k):
        cmd = [
            sys.executable, "-u", "scripts/run_fold.py",
            "--fold", str(k),
            "--threads", str(threads_per_proc),
        ]
        log_file = log_dir / f"fold{k}_run.log"
        log_print(f"[{time.strftime('%H:%M:%S')}] Launching Fold {k}...")
        log_print(f"  Command: {' '.join(cmd)}")
        log_print(f"  Log:     {log_file}")
        
        f_out = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.STDOUT, cwd=str(ROOT))
        active_processes[k] = (proc, f_out)
        log_print(f"  Fold {k} started with PID: {proc.pid}")

    # Initial launch up to max_parallel
    while len(active_processes) < max_parallel and len(folds_queue) > 0:
        k = folds_queue.pop(0)
        start_fold(k)

    log_print("\nInitial worker pool active. Monitoring execution...\n")

    # Monitor loop
    while len(active_processes) > 0:
        finished_folds = []
        for k, (proc, f_out) in active_processes.items():
            ret = proc.poll()
            if ret is not None:
                log_print(f"[{time.strftime('%H:%M:%S')}] Fold {k} finished with exit code {ret}!")
                f_out.close()
                finished_folds.append(k)

        for k in finished_folds:
            del active_processes[k]
            # If there are remaining folds in the queue, start the next one
            if len(folds_queue) > 0:
                next_k = folds_queue.pop(0)
                start_fold(next_k)

        time.sleep(5)

    log_print("\n" + "=" * 70)
    log_print("ALL 5 FOLDS COMPLETED SUCCESSFULLY!")
    log_print("=" * 70)
    manager_log.close()

if __name__ == "__main__":
    launch_folds()
