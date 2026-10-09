"""MegaDepth-1500 Dataset Structure and Integrity Check.

Validates that data/megadepth1500/ contains all expected metadata files,
calibrated pair definitions, and corresponding image files on disk.
"""

from pathlib import Path
import sys


def find_file(base_dir: Path, name: str) -> Path | None:
    """Locate a file with or without .txt extension."""
    candidates = [base_dir / name, base_dir / f"{name}.txt"]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main():
    print("=" * 60)
    print("MegaDepth-1500 Dataset Integrity Check")
    print("=" * 60)

    # 1. Robust project root determination
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    data_dir = project_root / "data" / "megadepth1500"

    print(f"Project root : {project_root}")
    print(f"Dataset root : {data_dir}\n")

    if not data_dir.exists():
        print(f"[FAIL] Dataset directory does not exist: {data_dir}")
        sys.exit(1)

    # 2. Check required files and directories
    required_items = {
        "pairs": find_file(data_dir, "pairs"),
        "pairs_calibrated": find_file(data_dir, "pairs_calibrated"),
        "views": find_file(data_dir, "views"),
        "images": data_dir / "images" if (data_dir / "images").is_dir() else None,
        "depths": data_dir / "depths" if (data_dir / "depths").is_dir() else None,
    }

    print("--- 1. Structure & Path Verification ---")
    all_exist = True
    for key, path in required_items.items():
        if path is not None and path.exists():
            rel_path = path.relative_to(project_root)
            kind = "Directory" if path.is_dir() else "File"
            print(f"  [OK]   {key:<18} -> {rel_path} ({kind})")
        else:
            print(f"  [MISS] {key:<18} -> NOT FOUND")
            all_exist = False

    if not all_exist:
        print("\n[FAIL] Missing required files/folders. Dataset incomplete.")
        sys.exit(1)

    pairs_calib_file = required_items["pairs_calibrated"]
    images_dir = required_items["images"]

    # 3. Read and validate pairs_calibrated
    print("\n--- 2. Calibrated Pairs Analysis ---")
    lines = [line.strip() for line in pairs_calib_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_pairs = len(lines)
    print(f"Total non-empty calibrated pairs found: {total_pairs}")

    if total_pairs == 0:
        print("[FAIL] pairs_calibrated file is empty.")
        sys.exit(1)

    # 4. Inspect the first calibrated pair format
    first_line = lines[0]
    first_tokens = first_line.split()
    num_fields = len(first_tokens)

    print("\n--- 3. First Pair Format Inspection ---")
    print(f"Field count in line 1: {num_fields} tokens")

    # Expected: img1 (1), img2 (1), K1 (9), K2 (9), R (9), T (3) = 32 tokens
    if num_fields == 32:
        img1_sample = first_tokens[0]
        img2_sample = first_tokens[1]
        k1_sample = first_tokens[2:11]
        k2_sample = first_tokens[11:20]
        r_sample = first_tokens[20:29]
        t_sample = first_tokens[29:32]

        print(f"  Image 1 path             : {img1_sample}")
        print(f"  Image 2 path             : {img2_sample}")
        print(f"  Intrinsics K1 (9 values) : {' '.join(k1_sample[:3])} ...")
        print(f"  Intrinsics K2 (9 values) : {' '.join(k2_sample[:3])} ...")
        print(f"  Rotation R    (9 values) : {' '.join(r_sample[:3])} ...")
        print(f"  Translation T (3 values) : {' '.join(t_sample)}")
        print("  [OK] Line 1 format matches standard 32-field calibrated pose layout.")
    else:
        print(f"  [WARN] Unexpected field count ({num_fields}). Expected 32.")

    # 5. Verify image file existence for all pairs
    print("\n--- 4. Image File & Folder Verification ---")
    referenced_folders = set()
    valid_pairs = 0
    missing_pairs = []

    for idx, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) < 2:
            missing_pairs.append((idx + 1, "invalid line format", ""))
            continue

        img1_rel, img2_rel = tokens[0], tokens[1]
        p1 = images_dir / img1_rel
        p2 = images_dir / img2_rel

        # Track referenced subfolders
        if "/" in img1_rel:
            referenced_folders.add(img1_rel.split("/")[0])
        if "/" in img2_rel:
            referenced_folders.add(img2_rel.split("/")[0])

        if p1.is_file() and p2.is_file():
            valid_pairs += 1
        else:
            missing_pairs.append((idx + 1, img1_rel, img2_rel))

    print(f"Referenced image folders in pairs : {sorted(list(referenced_folders))}")
    for folder in sorted(list(referenced_folders)):
        fpath = images_dir / folder
        status = "EXISTS" if fpath.is_dir() else "MISSING"
        print(f"  Folder '{folder}': {status} ({fpath})")

    print("\n--- 5. Dataset Validation Summary ---")
    print(f"Total calibrated pairs  : {total_pairs}")
    print(f"Valid image pairs       : {valid_pairs}")
    print(f"Missing image pairs     : {len(missing_pairs)}")

    if missing_pairs:
        print("\nFirst missing pairs (up to 5):")
        for line_num, i1, i2 in missing_pairs[:5]:
            print(f"  Line {line_num}: {i1} <-> {i2}")
        print("\n[FAIL] Some referenced image pairs are missing.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] 100% of referenced image files exist on disk.")
        print("[SUCCESS] MegaDepth-1500 dataset is verified and ready for evaluation.")


if __name__ == "__main__":
    main()
