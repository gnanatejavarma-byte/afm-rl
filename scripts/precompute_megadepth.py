import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2
from tqdm import tqdm
import pipeline_utils as pu
import megadepth as md
from megadepth_env import pair_stats

ap = argparse.ArgumentParser()
ap.add_argument("--presets", nargs="+", default=pu.PRESET_NAMES)
ap.add_argument("--stats", action="store_true", help="only compute pair statistics")
args = ap.parse_args()
presets = [p.replace("_", "+") for p in args.presets]

cv2.setNumThreads(2)
try:
    import torch
    torch.set_num_threads(2)
except ImportError:
    pass

pairs = md.list_pairs()
images = {}
for fname, i, j in pairs:
    for k in (i, j):
        images.setdefault(md.image_key(fname, k), (fname, k))
print(len(images), "unique images,", len(pairs), "pairs")

if args.stats:
    for fname, i, j in tqdm(pairs, desc="pair stats"):
        pair_stats(fname, i, j)
    sys.exit()

for preset in presets:
    t0, built = time.time(), 0
    for key, (fname, k) in tqdm(images.items(), desc=preset):
        if pu._disk_path(key, preset).exists():
            continue
        img, _ = md.load_image(fname, k)
        pu.get_pool(img, preset, key)
        built += 1
    print(f"{preset}: built {built} pools in {time.time() - t0:.0f}s")