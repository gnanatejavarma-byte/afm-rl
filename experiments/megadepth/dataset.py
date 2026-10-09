"""MegaDepth-1500 Dataset Loader.

Parses data/megadepth1500/pairs_calibrated.txt and yields calibrated image pairs
with camera intrinsics (K1, K2) and ground-truth relative pose (R_gt, T_gt).
Does NOT use or require depth maps.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np


class MegaDepthDataset:
    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            # Default: project_root / data / megadepth1500
            self.project_root = Path(__file__).resolve().parents[2]
            self.root_dir = self.project_root / "data" / "megadepth1500"
        else:
            self.root_dir = Path(root_dir)
            self.project_root = self.root_dir.parents[1]

        self.images_dir = self.root_dir / "images"
        self.calib_file = self._find_file("pairs_calibrated")
        if self.calib_file is None or not self.calib_file.is_file():
            raise FileNotFoundError(f"Calibrated pairs file not found under {self.root_dir}")

        self.pairs: List[Dict[str, Any]] = self._parse_pairs()

    def _find_file(self, name: str) -> Optional[Path]:
        candidates = [self.root_dir / name, self.root_dir / f"{name}.txt"]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _parse_pairs(self) -> List[Dict[str, Any]]:
        parsed = []
        lines = self.calib_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) != 32:
                continue

            img1_rel = tokens[0]
            img2_rel = tokens[1]
            k1_vals = [float(x) for x in tokens[2:11]]
            k2_vals = [float(x) for x in tokens[11:20]]
            r_vals = [float(x) for x in tokens[20:29]]
            t_vals = [float(x) for x in tokens[29:32]]

            K1 = np.array(k1_vals, dtype=np.float64).reshape(3, 3)
            K2 = np.array(k2_vals, dtype=np.float64).reshape(3, 3)
            R_gt = np.array(r_vals, dtype=np.float64).reshape(3, 3)
            T_gt = np.array(t_vals, dtype=np.float64).reshape(3, 1)

            parsed.append({
                "pair_id": idx,
                "img1_rel": img1_rel,
                "img2_rel": img2_rel,
                "img1_path": self.images_dir / img1_rel,
                "img2_path": self.images_dir / img2_rel,
                "K1": K1,
                "K2": K2,
                "R_gt": R_gt,
                "T_gt": T_gt,
            })
        return parsed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.pairs[idx]

    def load_images(self, idx: int, grayscale: bool = True) -> Dict[str, Any]:
        """Loads the pair metadata and image numpy arrays from disk."""
        item = self.pairs[idx]
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img1 = cv2.imread(str(item["img1_path"]), flag)
        img2 = cv2.imread(str(item["img2_path"]), flag)

        if img1 is None:
            raise FileNotFoundError(f"Could not load image: {item['img1_path']}")
        if img2 is None:
            raise FileNotFoundError(f"Could not load image: {item['img2_path']}")

        return {
            **item,
            "img1": img1,
            "img2": img2,
            "shape1": img1.shape[:2],
            "shape2": img2.shape[:2],
        }
