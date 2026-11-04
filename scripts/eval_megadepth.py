import sys, os, csv, argparse, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
import pipeline_utils as pu
import megadepth as md
from megadepth_env import MegaDepthEnv

FIXED_N = {"ORB": 1000, "SIFT": 2000, "AKAZE": 1000, "BRISK": 2000, "FAST+BRIEF": 1000,
           "KAZE": 4000, "GFTT+BRIEF": 1000, "GFTT+SIFT": 1000, "STAR+BRIEF": 100,
           "ORB+FREAK": 2000, "FAST+FREAK": 1000, "XFeat": 1000, "ALIKED": 500, "ORB+BEBLID": 4000}
COLS = ["preset", "file", "i", "j", "bucket", "fixed_n", "fixed_kp", "fixed_err",
        "best_kp", "best_err", "final_kp", "final_err", "steps"]

ap = argparse.ArgumentParser()
ap.add_argument("--presets", nargs="+", default=pu.PRESET_NAMES)
ap.add_argument("--max-pairs", type=int, default=0, help="0 = all 1500 pairs")
ap.add_argument("--model", default="checkpoints/level2_14preset_best/best_model.zip")
ap.add_argument("--out", default="logs/megadepth")
args = ap.parse_args()
presets = [p.replace("_", "+") for p in args.presets]
cv2.setNumThreads(1)
os.makedirs(args.out, exist_ok=True)

pairs = md.list_pairs()
if args.max_pairs:
    pairs = [pairs[x] for x in np.linspace(0, len(pairs) - 1, args.max_pairs).astype(int)]

env = MegaDepthEnv()
model = PPO.load(args.model, device="cpu")
checked = False

for preset in presets:
    path = os.path.join(args.out, f"eval_{preset.replace('+', '_')}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {(r["file"], int(r["i"]), int(r["j"])) for r in csv.DictReader(fh)}
    fresh = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if fresh:
        w.writeheader()
    todo = [p for p in pairs if p not in done]
    errors = 0
    for pair in tqdm(todo, desc=preset):
        try:
            obs, _ = env.reset_to(pair, preset)
            if not checked:
                assert obs.shape == (34,), obs.shape
                print("observation shape OK:", obs.shape)
                checked = True
            steps = 0
            while True:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(int(a))
                steps += 1
                if term or trunc:
                    break
            final_kp, final_err = env.effective_kp, env.last_m["pose_err"]
            best_kp, best_err = info["best_n_kp"], info["best_corner_err"]
            env.n_kp = FIXED_N[preset]
            env._run_and_eval()
            fixed_kp, fixed_err = env.effective_kp, env.last_m["pose_err"]
            w.writerow(dict(preset=preset, file=pair[0], i=pair[1], j=pair[2],
                            bucket=md.bucket(pair[0]), fixed_n=FIXED_N[preset],
                            fixed_kp=fixed_kp, fixed_err=fixed_err, best_kp=best_kp,
                            best_err=best_err, final_kp=final_kp, final_err=final_err, steps=steps))
            fh.flush()
        except Exception:
            errors += 1
            if errors <= 3:
                traceback.print_exc()
    fh.close()
    print(f"{preset}: done, {errors} errors")