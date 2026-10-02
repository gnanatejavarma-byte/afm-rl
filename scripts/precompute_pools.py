import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tqdm import tqdm
from hpatches import load_split, load_pair
import pipeline_utils as pu

split = load_split()
seqs = split["train"] + split["val"]   # test split stays untouched until Phase 6

jobs = [(seq, idx) for seq in seqs for idx in range(1, 7)]
print(f"{len(jobs)} images x {len(pu.PRESET_NAMES)} presets = "
      f"{len(jobs) * len(pu.PRESET_NAMES)} pools to build (skips ones already on disk)")

t0 = time.perf_counter()
built = 0
for seq, idx in tqdm(jobs, desc="images"):
    img1, img2, _ = load_pair(seq, max(idx, 2))     # load_pair needs img1 + imgK; reuse img1 for idx==1
    img = img1 if idx == 1 else img2
    for preset in pu.PRESET_NAMES:
        path = pu._disk_path((seq, idx), preset)
        if path.exists():
            continue
        _, dt = pu.get_pool(img, preset, key=(seq, idx))
        built += 1

print(f"\nBuilt {built} new pools in {time.perf_counter()-t0:.1f}s "
      f"(rest were already cached on disk)")