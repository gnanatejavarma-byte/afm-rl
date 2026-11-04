import sys, os, csv, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
from tqdm import tqdm
import pipeline_utils as pu
import megadepth as md
from megadepth_env import MegaDepthEnv

GRID = [500, 1000, 2000, 4000]
ap = argparse.ArgumentParser()
ap.add_argument("--presets", nargs="+", default=pu.PRESET_NAMES)
ap.add_argument("--max-pairs", type=int, default=0)
ap.add_argument("--out", default="logs/megadepth")
args = ap.parse_args()
presets = [p.replace("_", "+") for p in args.presets]
cv2.setNumThreads(1)
os.makedirs(args.out, exist_ok=True)
pairs = md.list_pairs()
if args.max_pairs:
    pairs = [pairs[x] for x in np.linspace(0, len(pairs) - 1, args.max_pairs).astype(int)]
env = MegaDepthEnv()
cols = ["preset", "file", "i", "j"] + [f"err_{n}" for n in GRID] + [f"kp_{n}" for n in GRID]
for preset in presets:
    path = os.path.join(args.out, f"grid_{preset.replace('+', '_')}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {(r["file"], int(r["i"]), int(r["j"])) for r in csv.DictReader(fh)}
    fresh = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if fresh:
        w.writeheader()
    for pair in tqdm([p for p in pairs if p not in done], desc=preset):
        env.reset_to(pair, preset)
        row = dict(preset=preset, file=pair[0], i=pair[1], j=pair[2])
        for n in GRID:
            env.n_kp = n
            env._run_and_eval()
            row[f"err_{n}"] = env.last_m["pose_err"]
            row[f"kp_{n}"] = env.effective_kp
        w.writerow(row)
        fh.flush()
    fh.close()