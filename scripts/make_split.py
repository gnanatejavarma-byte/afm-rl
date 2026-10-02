import argparse, json, random, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/hpatches")
ap.add_argument("--out", default="configs/split.json")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--force", action="store_true", help="overwrite an existing split")
args = ap.parse_args()

out = Path(args.out)
if out.exists() and not args.force:
    sys.exit(f"{out} already exists. Not overwriting (this protects the test split). Use --force only if you really mean it.")

root = Path(args.root)
names = sorted({p.name for p in root.rglob("*")
                if p.is_dir() and p.name[:2] in ("i_", "v_")})
assert len(names) == 116, f"expected 116 sequences, found {len(names)}"

FRAC_TRAIN, FRAC_VAL = 0.70, 0.15   # remainder (~0.15) is test
rng = random.Random(args.seed)
split = {"train": [], "val": [], "test": []}

for prefix in ("i_", "v_"):
    group = [n for n in names if n.startswith(prefix)]
    rng.shuffle(group)
    n_train = round(FRAC_TRAIN * len(group))
    n_val = round(FRAC_VAL * len(group))
    split["train"] += group[:n_train]
    split["val"] += group[n_train:n_train + n_val]
    split["test"] += group[n_train + n_val:]

for k in split:
    split[k].sort()

# safety checks: disjoint and complete
all_assigned = split["train"] + split["val"] + split["test"]
assert len(all_assigned) == len(set(all_assigned)) == 116

out.parent.mkdir(parents=True, exist_ok=True)
payload = {"seed": args.seed, "ratios": "70/15/15",
           "note": "split by sequence, stratified by i_/v_; pairs = ref image 1 vs 2..6",
           **split}
out.write_text(json.dumps(payload, indent=2))

print(f"Saved {out}")
for k in ("train", "val", "test"):
    n_i = sum(n.startswith("i_") for n in split[k])
    n_v = sum(n.startswith("v_") for n in split[k])
    print(f"{k:5s}: {len(split[k]):3d} sequences ({n_i} illum, {n_v} view) = {len(split[k]) * 5} pairs")