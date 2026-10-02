import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, pandas as pd
from tqdm import tqdm
from hpatches import load_split, load_pair
import pipeline_utils as pu
from metrics import evaluate_matches

cv2.setRNGSeed(0)                       # reproducible RANSAC
val = load_split()["val"]
seqs = [s for s in val if s.startswith("i_")][:3] + [s for s in val if s.startswith("v_")][:3]
GRID = [100, 500, 1000, 2000, 4000]
PAIRS = [(s, k) for s in seqs for k in (3, 5)]

rows = []
for seq, k in tqdm(PAIRS, desc="pairs"):
    img1, img2, H = load_pair(seq, k)
    for p in pu.PRESET_NAMES:
        for n in GRID:
            r = pu.run_pipeline(img1, img2, p, n, matcher="BF",
                                key1=(seq, 1), key2=(seq, k))
            m = evaluate_matches(r["pts1"], r["pts2"], H, img1.shape)
            rows.append(dict(preset=p, N=n, kind=seq[:1], seq=seq, k=k, **m))

df = pd.DataFrame(rows)
os.makedirs("logs", exist_ok=True)
df.to_csv("logs/accuracy_test.csv", index=False)

pd.set_option("display.width", 200)
order = pu.PRESET_NAMES
for col, title in [("n_correct", "mean number of CORRECT matches"),
                   ("precision", "mean precision (correct / matches)"),
                   ("ransac_ratio", "mean RANSAC inlier ratio"),
                   ("h_ok", "homography success rate (corner error < 3 px)")]:
    t = df.pivot_table(index="preset", columns="N", values=col, aggfunc="mean").loc[order]
    print(f"\n--- {title} ---")
    print(t.round(2 if col != "n_correct" else 0).to_string())

print("\n--- homography success rate by scene type, N=1000 ---")
print(df[df.N == 1000].pivot_table(index="preset", columns="kind", values="h_ok",
                                   aggfunc="mean").loc[order].round(2).to_string())
print("\nSaved logs/accuracy_test.csv")