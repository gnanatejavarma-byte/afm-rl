import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/hpatches")
seq_dirs = sorted({p for p in root.rglob("*") if p.is_dir() and p.name[:2] in ("i_", "v_")})

illum = [d for d in seq_dirs if d.name.startswith("i_")]
view = [d for d in seq_dirs if d.name.startswith("v_")]
print(f"Found {len(seq_dirs)} sequences: {len(illum)} illumination (i_), {len(view)} viewpoint (v_)")

problems = []
pairs = 0
for d in seq_dirs:
    imgs_ok = all((d / f"{k}.ppm").exists() for k in range(1, 7))
    hs_ok = all((d / f"H_1_{k}").exists() for k in range(2, 7))
    if imgs_ok and hs_ok:
        pairs += 5
    else:
        problems.append(d.name)

print(f"Complete pairs (ref image vs. image 2..6): {pairs}")
print("Sequences with missing files:", problems if problems else "none")
if seq_dirs:
    print("Example location:", seq_dirs[0])

ok = len(seq_dirs) == 116 and pairs == 580 and not problems
print("RESULT:", "dataset looks correct" if ok else "MISMATCH, see above")