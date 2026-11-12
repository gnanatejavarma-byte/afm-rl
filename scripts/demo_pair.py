"""
scripts/demo_pair.py

Interactive Live Demo for AFM-RL (Level 1 Preset Selection + Level 2 Keypoint Budgeting).
Allows evaluating on:
1. Any two custom user images via file path (--img1, --img2)
2. Interactive Windows file picker dialog (--gui or default when no args passed)
3. Benchmark dataset samples (--sample hpatches or --sample megadepth)

Outputs:
- Level 1 Chosen Feature Preset (e.g. SIFT, ALIKED, etc.)
- Step-by-step Level 2 decision trajectory (e.g. 500 -> 1000 -> 1500 -> STOP)
- Final Keypoint Budget decided by RL agent
- Total matches, RANSAC inliers, inlier ratio, and runtime
- Visual side-by-side matching plot saved to demo_output/matched_result.png
"""

import sys
import os
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Set up paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline_utils as pu
import image_stats as ims
from metrics import reward_accuracy

L1_MODEL_PATH = ROOT / "checkpoints" / "level1_best" / "best_model.zip"
L2_MODEL_PATH = ROOT / "checkpoints" / "level2_14preset_best" / "best_model.zip"
STEP_SIZES = [500, 1000, -500, None]  # 0: +500, 1: +1000, 2: -500, 3: STOP
MAX_STEPS = 15
DEFAULT_LAMBDA = 0.1
MATCH_NORM = 1000.0


def select_files_via_dialog():
    """Opens a native Windows GUI file picker for image 1 and image 2."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        
        print("\nOpening file dialog: Please select Image 1...")
        img1_path = filedialog.askopenfilename(
            parent=root,
            title="Select Image 1",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.ppm"), ("All files", "*.*")]
        )
        if not img1_path:
            print("Selection cancelled for Image 1.")
            root.destroy()
            return None, None
            
        print(f"Selected Image 1: {img1_path}")
        print("Opening file dialog: Please select Image 2...")
        root.update()
        img2_path = filedialog.askopenfilename(
            parent=root,
            title="Select Image 2",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.ppm"), ("All files", "*.*")]
        )
        if not img2_path:
            print("Selection cancelled for Image 2.")
            root.destroy()
            return None, None

        print(f"Selected Image 2: {img2_path}\n")
        root.destroy()
        return img1_path, img2_path
    except Exception as e:
        print(f"GUI file dialog unavailable ({e}). Please specify --img1 and --img2 on CLI.")
        return None, None


def load_and_preprocess_image(path, max_dim=1200):
    """Loads image, converts to uint8 grayscale, and resizes if larger than max_dim."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image from: {path}")
    
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_gray


def run_demo(img1_path, img2_path, l1_path=L1_MODEL_PATH, l2_path=L2_MODEL_PATH, lam=DEFAULT_LAMBDA, show_plot=True):
    print("=" * 70)
    print("        AFM-RL: HIERARCHICAL MATCHING INFERENCE DEMO")
    print("=" * 70)
    
    # 1. Load models
    print(f"[*] Loading Level 1 model: {Path(l1_path).name}")
    l1_model = PPO.load(str(l1_path), device="cpu")
    print(f"[*] Loading Level 2 model: {Path(l2_path).name}")
    l2_model = PPO.load(str(l2_path), device="cpu")

    # 2. Load images
    print(f"\n[*] Image 1: {img1_path}")
    print(f"[*] Image 2: {img2_path}")
    bgr1, gray1 = load_and_preprocess_image(img1_path)
    bgr2, gray2 = load_and_preprocess_image(img2_path)
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    print(f"    Loaded Image 1: {w1}x{h1} px | Image 2: {w2}x{h2} px")

    # 3. Compute 9-dim visual statistics
    print("\n" + "-" * 70)
    print("STEP 1: LEVEL 1 (PIPELINE SELECTION)")
    print("-" * 70)
    t0 = time.perf_counter()
    raw_stats = ims.compute_pair_stats(gray1, gray2)
    norm_stats = ims.normalize_stats(raw_stats)
    t_stats = (time.perf_counter() - t0) * 1000

    print(f"Visual Pair Statistics (computed in {t_stats:.1f}ms):")
    stat_names = ["mean_brightness", "rms_contrast", "laplacian_sharpness",
                  "entropy", "colorfulness", "aspect_ratio",
                  "brightness_ratio", "contrast_ratio", "sharpness_ratio"]
    for name, val, n_val in zip(stat_names, raw_stats, norm_stats):
        print(f"  - {name:<22}: raw = {val:8.2f} | normalized = {n_val:6.2f}")

    # Predict Level 1 Preset
    action_l1, _ = l1_model.predict(norm_stats, deterministic=True)
    chosen_preset = pu.PRESET_NAMES[int(action_l1)]
    print(f"\n>>> LEVEL 1 DECISION: Preset = [{chosen_preset.upper()}] (action index {int(action_l1)}) <<<")

    # 4. Level 2 Adaptive Keypoint Budgeting Rollout
    print("\n" + "-" * 70)
    print("STEP 2: LEVEL 2 (ADAPTIVE KEYPOINT BUDGETING)")
    print("-" * 70)
    print(f"Initial Keypoint Budget: N = {pu.N_START}")
    print("Starting sequential RL budgeting loop...\n")

    n_presets = len(pu.PRESET_NAMES)
    preset_idx = pu.PRESET_NAMES.index(chosen_preset)
    onehot = np.zeros(n_presets, dtype=np.float32)
    onehot[preset_idx] = 1.0

    n_kp = pu.N_START
    steps = 0
    prev_utility = 0.0
    prev_acc = 0.0
    delta_acc = 0.0
    delta_utility = 0.0
    best_n_kp = n_kp
    best_utility = -float("inf")
    best_result = None

    history = []

    while steps < MAX_STEPS:
        steps += 1
        
        # Run pipeline at current budget
        r = pu.run_pipeline(gray1, gray2, chosen_preset, n_kp, matcher="BF", ratio=0.75)
        n_matches = r["n_matches"]
        effective_kp = r["n_kp"]
        pool_size = min(r["pool_n1"], r["pool_n2"])

        # Compute RANSAC inliers & homography
        inlier_ratio = 0.0
        n_inliers = 0
        inlier_mask = None
        H_est = None
        corner_err = float("inf")

        if n_matches >= 4:
            H_est, mask = cv2.findHomography(r["pts1"], r["pts2"], cv2.RANSAC, 3.0)
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                n_inliers = int(np.sum(inlier_mask))
                inlier_ratio = n_inliers / float(n_matches)

        # Accuracy estimate based on inlier ratio and match count stability
        acc_est = min(1.0, (n_inliers / 150.0)) * inlier_ratio
        utility = acc_est - lam * (effective_kp / pu.N_MAX)

        if steps == 1:
            prev_acc = acc_est
            prev_utility = utility
            best_utility = utility
            best_n_kp = effective_kp
            best_result = (r, n_inliers, inlier_ratio, inlier_mask)
        else:
            delta_acc = acc_est - prev_acc
            delta_utility = utility - prev_utility
            if utility > best_utility:
                best_utility = utility
                best_n_kp = effective_kp
                best_result = (r, n_inliers, inlier_ratio, inlier_mask)

        is_plateau = float(effective_kp >= pool_size)
        gap_from_best = utility - best_utility

        # Construct 34-dim observation vector
        extra = np.array([
            n_kp / pu.N_MAX,
            effective_kp / pu.N_MAX,
            pool_size / pu.N_MAX,
            acc_est,
            inlier_ratio,
            min(n_matches, MATCH_NORM) / MATCH_NORM,
            steps / MAX_STEPS,
            delta_acc,
            delta_utility,
            is_plateau,
            gap_from_best
        ], dtype=np.float32)
        obs = np.concatenate([norm_stats, onehot, extra]).astype(np.float32)

        # Query Level 2 Policy
        action_l2, _ = l2_model.predict(obs, deterministic=True)
        act_idx = int(action_l2)
        delta = STEP_SIZES[act_idx]
        act_name = "STOP" if delta is None else f"{delta:+d}"

        history.append({
            "step": steps,
            "budget": n_kp,
            "effective": effective_kp,
            "matches": n_matches,
            "inliers": n_inliers,
            "inlier_ratio": inlier_ratio,
            "action": act_name
        })

        print(f"  [Step {steps:02d}] Keypoint Budget: {n_kp:4d} | Matches: {n_matches:4d} | "
              f"Inliers: {n_inliers:4d} ({inlier_ratio*100:4.1f}%) | Action -> {act_name}")

        prev_acc = acc_est
        prev_utility = utility

        if delta is None:  # STOP chosen by RL agent
            print(f"\n>>> Level 2 Agent issued STOP signal at Step {steps}! <<<")
            break

        # Apply action delta
        n_kp = int(np.clip(n_kp + delta, pu.N_MIN, pu.N_MAX))

    final_chosen_budget = n_kp
    print("\n" + "=" * 70)
    print("                    FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"  * Level 1 Selected Preset : {chosen_preset.upper()}")
    print(f"  * Level 2 Stopped At Budget: {final_chosen_budget} keypoints")
    print(f"  * Total Rollout Steps     : {steps}")
    last_h = history[-1]
    print(f"  * Total Matches Found     : {last_h['matches']}")
    print(f"  * Verified RANSAC Inliers : {last_h['inliers']} ({last_h['inlier_ratio']*100:.1f}%)")
    print("=" * 70)

    # 5. Visualization
    r_final, inliers_final, ratio_final, mask_final = best_result if best_result is not None else (r, n_inliers, inlier_ratio, inlier_mask)
    pts1 = r_final["pts1"]
    pts2 = r_final["pts2"]
    
    out_dir = ROOT / "demo_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "matched_result.png"

    # Convert bgr to rgb for matplotlib
    rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 7))
    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    canvas = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = rgb1
    canvas[:h2, w1:w1+w2] = rgb2

    ax.imshow(canvas)
    ax.axis("off")

    # Draw subset of inlier matches for clean visualization (up to 80 matches)
    if mask_final is not None and len(pts1) > 0:
        inlier_pts1 = pts1[mask_final]
        inlier_pts2 = pts2[mask_final]
        n_draw = min(len(inlier_pts1), 80)
        draw_indices = np.random.choice(len(inlier_pts1), n_draw, replace=False) if len(inlier_pts1) > 80 else range(len(inlier_pts1))
        
        for idx in draw_indices:
            x1, y1 = inlier_pts1[idx]
            x2, y2 = inlier_pts2[idx]
            ax.plot([x1, x2 + w1], [y1, y2], color="lime", alpha=0.6, linewidth=1.0)
            ax.scatter([x1], [y1], color="yellow", s=10)
            ax.scatter([x2 + w1], [y2], color="yellow", s=10)

    title_str = (
        f"AFM-RL Hierarchical Matching Result\n"
        f"Level 1 Preset: {chosen_preset.upper()}  |  "
        f"Level 2 Budget: {final_chosen_budget} Keypoints  |  "
        f"Inliers: {inliers_final}/{len(pts1)} ({ratio_final*100:.1f}%)"
    )
    ax.set_title(title_str, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(str(out_img_path), dpi=150, bbox_inches="tight")
    print(f"\n[+] Visual matching plot saved to: {out_img_path}")
    
    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
            
    return {
        "chosen_preset": chosen_preset,
        "final_budget": final_chosen_budget,
        "steps": steps,
        "matches": last_h["matches"],
        "inliers": last_h["inliers"],
        "output_image": str(out_img_path)
    }


def main():
    parser = argparse.ArgumentParser(description="AFM-RL Demo on Arbitrary Image Pairs")
    parser.add_argument("--img1", type=str, default=None, help="Path to first image")
    parser.add_argument("--img2", type=str, default=None, help="Path to second image")
    parser.add_argument("--gui", action="store_true", help="Launch native file dialog to choose images")
    parser.add_argument("--sample", type=str, choices=["hpatches", "megadepth"], default=None,
                        help="Run on a sample benchmark pair")
    parser.add_argument("--no_show", action="store_true", help="Do not open popup plot window")
    args = parser.parse_args()

    img1_path, img2_path = args.img1, args.img2

    # Check sample mode
    if args.sample == "hpatches":
        img1_path = str(ROOT / "data" / "hpatches" / "i_ajuntament" / "1.ppm")
        img2_path = str(ROOT / "data" / "hpatches" / "i_ajuntament" / "2.ppm")
    elif args.sample == "megadepth":
        import megadepth as md
        pairs = md.list_pairs()
        if len(pairs) > 0:
            fname, i, j = pairs[0]
            bgr1, bgr2, _ = md.load_pair(fname, i, j)
            tmp_dir = ROOT / "demo_output" / "megadepth_sample"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            img1_path = str(tmp_dir / "img1.jpg")
            img2_path = str(tmp_dir / "img2.jpg")
            cv2.imwrite(img1_path, bgr1)
            cv2.imwrite(img2_path, bgr2)

    # If images still not provided, check demo_input folder
    if img1_path is None or img2_path is None:
        demo_input_dir = ROOT / "demo_input"
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"}
        input_files = sorted([p for p in demo_input_dir.glob("*.*") if p.suffix.lower() in valid_exts]) if demo_input_dir.exists() else []
        
        if len(input_files) >= 2 and not args.gui:
            img1_path = str(input_files[0])
            img2_path = str(input_files[1])
            print(f"[i] Found images in demo_input/: using '{input_files[0].name}' and '{input_files[1].name}'")
        else:
            # Fall back to native GUI dialog
            print("Opening Windows file picker dialog...")
            img1_path, img2_path = select_files_via_dialog()

    if not img1_path or not img2_path:
        print("\nError: Please provide two images using either:")
        print("  1. CLI: python scripts/demo_pair.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg")
        print("  2. Sample: python scripts/demo_pair.py --sample hpatches")
        print("  3. Drag images into demo_input/ folder (demo_input/image1.jpg and demo_input/image2.jpg)")
        print("  4. GUI Dialog: python scripts/demo_pair.py --gui")
        sys.exit(1)

    run_demo(img1_path, img2_path, show_plot=not args.no_show)


if __name__ == "__main__":
    main()
