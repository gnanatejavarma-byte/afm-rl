"""Dataset split generation for MegaDepth-1500 Level-2 Training.

Generates deterministic, non-overlapping train/val/test split:
- Train: 1050 pairs (70%)
- Validation: 225 pairs (15%)
- Test: 225 pairs (15%)
Total: 1500 pairs.

Saves: experiments/megadepth_training/split_manifest.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
MEGADEPTH_EXP = PROJECT_ROOT / "experiments" / "megadepth"

for p in [str(PROJECT_ROOT / "src"), str(MEGADEPTH_EXP), str(EXP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import MegaDepthDataset

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def generate_split_manifest():
    dataset = MegaDepthDataset()
    total_pairs = len(dataset)
    assert total_pairs == 1500, f"Expected 1500 pairs, got {total_pairs}"

    rng = np.random.RandomState(SEED)
    shuffled_indices = np.arange(total_pairs)
    rng.shuffle(shuffled_indices)

    n_train = int(total_pairs * TRAIN_RATIO)  # 1050
    n_val = int(total_pairs * VAL_RATIO)      # 225
    n_test = total_pairs - n_train - n_val    # 225

    train_idx = set(shuffled_indices[:n_train])
    val_idx = set(shuffled_indices[n_train:n_train + n_val])
    test_idx = set(shuffled_indices[n_train + n_val:])

    # Strict overlap checks
    assert len(train_idx) == 1050
    assert len(val_idx) == 225
    assert len(test_idx) == 225
    assert len(train_idx.intersection(val_idx)) == 0, "Train and Val overlap!"
    assert len(train_idx.intersection(test_idx)) == 0, "Train and Test overlap!"
    assert len(val_idx.intersection(test_idx)) == 0, "Val and Test overlap!"

    rows = []
    for i in range(total_pairs):
        pair = dataset.pairs[i]
        if i in train_idx:
            split = "train"
        elif i in val_idx:
            split = "val"
        else:
            split = "test"

        rows.append({
            "pair_idx": i,
            "split": split,
            "img1_rel": pair["img1_rel"],
            "img2_rel": pair["img2_rel"],
        })

    df = pd.DataFrame(rows)
    out_path = EXP_DIR / "split_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved split manifest to {out_path}")
    print(f"     Train: {len(train_idx)} pairs ({len(train_idx)/total_pairs*100:.1f}%)")
    print(f"     Val:   {len(val_idx)} pairs ({len(val_idx)/total_pairs*100:.1f}%)")
    print(f"     Test:  {len(test_idx)} pairs ({len(test_idx)/total_pairs*100:.1f}%)")
    return df


if __name__ == "__main__":
    generate_split_manifest()
