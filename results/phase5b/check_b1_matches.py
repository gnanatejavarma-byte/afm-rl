import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'src')
import megadepth as md
import pipeline_utils as pu

MATCH_DIR = Path("results/phase4b/megadepth_matches")
POOL_CACHE_DIR = Path("data/pool_cache_megadepth")

needed_configs = [
    ("ALIKED", 500),
    ("ALIKED", 1000),
    ("ALIKED", 2000),
    ("SIFT", 500),
    ("SIFT", 1000),
    ("SIFT", 2000),
    ("SIFT", 4000),
]

pairs = md.list_pairs()
print(f"Checking match files for {len(pairs)} pairs across {len(needed_configs)} configs (total {len(pairs)*len(needed_configs)} files)...")

missing = []
existing_count = 0

for fname, i, j in pairs:
    for preset, N in needed_configs:
        clean_preset = preset.replace("+", "_")
        match_f = MATCH_DIR / f"{fname}_{i}_{j}_{clean_preset}_N{N}.npz"
        if match_f.exists():
            existing_count += 1
        else:
            missing.append((fname, i, j, preset, N))

print(f"Existing match files found: {existing_count}/{len(pairs)*len(needed_configs)}")
print(f"Missing match files: {len(missing)}")

if missing:
    print(f"Regenerating {len(missing)} missing match files from cached pools...")
    regenerated_count = 0
    for fname, i, j, preset, N in missing:
        img0, img1, K0, K1, T_gt = md.load_pair(fname, i, j)
        d = md._scene(fname)
        path0 = str(d["image_paths"][i])
        path1 = str(d["image_paths"][j])
        
        # Load cached pools
        pool0 = np.load(POOL_CACHE_DIR / preset / f"{path0.replace('/', '_')}.npz") if (POOL_CACHE_DIR / preset / f"{path0.replace('/', '_')}.npz").exists() else None
        # Use helper from megadepth_eval_env
        from megadepth_eval_env import get_megadepth_pool
        p0 = get_megadepth_pool(img0, preset, path0)
        p1 = get_megadepth_pool(img1, preset, path1)
        
        n0 = min(N, len(p0["pts"]))
        n1 = min(N, len(p1["pts"]))
        d0 = p0["desc"][:n0] if n0 else None
        d1 = p1["desc"][:n1] if n1 else None
        
        good = pu.match_ratio(d0, d1, preset, "BF", 0.75)
        if good:
            pts0 = np.float32([p0["pts"][m.queryIdx] for m in good])
            pts1 = np.float32([p1["pts"][m.trainIdx] for m in good])
        else:
            pts0 = pts1 = np.zeros((0, 2), dtype=np.float32)
            
        clean_preset = preset.replace("+", "_")
        match_f = MATCH_DIR / f"{fname}_{i}_{j}_{clean_preset}_N{N}.npz"
        np.savez_compressed(match_f, pts0=pts0, pts1=pts1)
        regenerated_count += 1
        if regenerated_count % 100 == 0:
            print(f"Regenerated {regenerated_count}/{len(missing)} files...")
    print(f"Successfully regenerated all {regenerated_count} missing match files.")
else:
    print("All required match files exist on disk! No regeneration needed.")

# Write summary CSV
df_summary = pd.DataFrame([{
    "n_pairs": len(pairs),
    "total_needed_files": len(pairs) * len(needed_configs),
    "existing_files": existing_count,
    "missing_files": len(missing),
    "regenerated_files": len(missing) if missing else 0
}])
df_summary.to_csv("results/phase5b/b1_match_inventory_summary.csv", index=False)
print("Saved summary to results/phase5b/b1_match_inventory_summary.csv")
