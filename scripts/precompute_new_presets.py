"""
precompute_new_presets.py
Pre-builds and disk-caches keypoint pools for the 3 new presets only
(XFeat, ALIKED, ORB+BEBLID) across the train+val split.

The test split is untouched.
Run this once before starting training.  Already-cached files are skipped.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tqdm import tqdm
from hpatches import load_split, load_pair
import pipeline_utils as pu

# Only the 3 new presets — existing OpenCV pools stay as-is
NEW_PRESETS = ["XFeat", "ALIKED", "ORB+BEBLID"]
for p in NEW_PRESETS:
    assert p in pu.PRESET_NAMES, f"{p} not found in PRESET_NAMES — check pipeline_utils.py"

split  = load_split()
seqs   = split["train"] + split["val"]   # test split is NEVER touched here
# Every image: index 1 (reference) plus images 2-6 (query)
jobs   = [(seq, idx) for seq in seqs for idx in range(1, 7)]

print(f"Train+val sequences : {len(seqs)}")
print(f"Images to cache     : {len(jobs)}")
print(f"New presets         : {NEW_PRESETS}")
print(f"Total pool files    : {len(jobs) * len(NEW_PRESETS)}")
print()

for preset in NEW_PRESETS:
    print(f"--- Preset: {preset} ---")
    t_preset = time.perf_counter()
    built = 0
    skipped = 0

    for seq, idx in tqdm(jobs, desc=preset, unit="img"):
        path = pu._disk_path((seq, idx), preset)
        if path.exists():
            skipped += 1
            continue

        # load_pair needs two images; for idx==1 reuse img1 as itself
        img1, img2, _ = load_pair(seq, max(idx, 2))
        img = img1 if idx == 1 else img2

        _, dt = pu.get_pool(img, preset, key=(seq, idx))
        built += 1

    elapsed = time.perf_counter() - t_preset
    print(f"   Built {built} new pools, skipped {skipped} cached  "
          f"({elapsed:.1f}s total)\n")

print("Done.  All new preset pools are cached on disk.")
print("You can now run:  python scripts/train_level2.py")
