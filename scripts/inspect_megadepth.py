import os, glob, numpy as np
from PIL import Image

ROOT = "data/megadepth/megadepth_test_1500"
INFO = "data/megadepth/megadepth_test_1500_scene_info"
print(open(os.path.join(INFO, "megadepth_test_1500.txt")).read())

total, missing, used = 0, 0, set()
for f in sorted(glob.glob(os.path.join(INFO, "*.npz"))):
    d = np.load(f, allow_pickle=True)
    pairs = d["pair_infos"]
    total += len(pairs)
    for (i, j), *_ in pairs:
        for k in (i, j):
            p = os.path.join(ROOT, str(d["image_paths"][k]))
            used.add(p)
            missing += not os.path.exists(p)
    print(os.path.basename(f), "keys:", list(d.keys()), "pairs:", len(pairs))

print("total pairs:", total, "| unique images used:", len(used), "| missing files:", missing)

d = np.load(sorted(glob.glob(os.path.join(INFO, "*.npz")))[0], allow_pickle=True)
(i, j), *rest = d["pair_infos"][0]
print("first pair:", i, j, rest[:1])
print("paths:", d["image_paths"][i], d["image_paths"][j])
print("img size:", Image.open(os.path.join(ROOT, str(d["image_paths"][i]))).size)
print("K0:\n", d["intrinsics"][i])
print("pose0:\n", d["poses"][i])