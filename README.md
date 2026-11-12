# AFM-RL: Adaptive Feature Matching Using Hierarchical Reinforcement Learning

An intelligent, two-level hierarchical reinforcement learning system that dynamically selects both the optimal feature matching pipeline and the minimal keypoint budget per image pair to optimize matching accuracy vs. computational cost.

---

## 1. Problem Statement
Traditional computer vision pipelines (such as SIFT, ORB, AKAZE, or learned matchers) use static algorithm choices and fixed keypoint budgets (e.g., always 2,000 or 4,000 points) regardless of image difficulty:
- **Easy, textured scenes** waste computation and battery by extracting thousands of redundant keypoints.
- **Challenging, dark, blurry, or low-texture scenes** fail when assigned weak classical extractors or insufficient keypoint density.

**AFM-RL** replaces static configurations with an adaptive, two-level hierarchical RL decision system:
1. **Level 1 (Pipeline Selection Agent):** Inspects pre-match visual characteristics (sharpness, entropy, contrast, illumination) and selects the best feature matching pipeline out of **14 presets**.
2. **Level 2 (Keypoint Budgeting Agent):** Sequentially adjusts the keypoint budget ($N$) with actions `+500`, `+1000`, `-500`, or `STOP` until peak matching utility is reached with minimal keypoints.

---

## 2. Two-Level Hierarchical Architecture

```
                                Image Pair (I₁, I₂)
                                         │
                                         ▼
                           Extract 9-Dim Visual Statistics
                     (Entropy, Contrast, Sharpness, Illumination)
                                         │
                                         ▼
                     LEVEL 1: Pipeline Selection Agent (Contextual Bandit)
                     └── Selects 1 of 14 Feature Presets (Classical / Learned)
                                         │
                                         ▼
                     LEVEL 2: Keypoint Budgeting Agent (Sequential PPO)
                     └── Dynamically tunes keypoint budget: [+500, +1000, -500, STOP]
                                         │
                                         ▼
                     Final Inlier Matches + Maximal AUC with Minimal Keypoint Cost
```

### Hierarchy Comparison

| Feature | Level 1: Pipeline Selection Agent | Level 2: Keypoint Budgeting Agent |
| :--- | :--- | :--- |
| **Role** | Selects optimal feature detector/descriptor preset | Dynamically tunes keypoint budget ($N$) |
| **Decision Type** | Single-shot contextual decision (Contextual Bandit) | Sequential multi-step decision ($T_{\text{max}} = 15$ steps) |
| **Algorithm** | **PPO (Contextual Bandit, $\gamma = 0$)** | **PPO (Proximal Policy Optimization)** |
| **Action Space** | Discrete(14): Selects 1 of 14 presets | Discrete(4): `+500`, `+1000`, `-500`, `STOP` |
| **Observation** | 9-dim visual pair statistics | **34-dim vector** (9 visual stats + 14 one-hot preset + 11 dynamic indicators) |
| **Reward Signal** | Final utility achieved by the frozen downstream Level 2 agent | Potential-based utility delta ($\Delta \mathcal{U} - \text{step\_penalty}$) |
| **Status** | **Trained & Evaluated** (`checkpoints/level1_best/`) | **Trained & Evaluated** (`checkpoints/level2_14preset_best/`) |

---

## 3. Supported 14 Feature Presets

AFM-RL operates across **14 diverse feature matching presets**, combining classical hand-crafted methods and modern deep-learned models:

| # | Preset Name | Detector | Descriptor | Metric / Type | Framework |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `ORB` | ORB | ORB | Hamming (Binary) | OpenCV |
| 2 | `SIFT` | SIFT | SIFT | L2 (Float) | OpenCV |
| 3 | `AKAZE` | AKAZE | AKAZE | Hamming (Binary) | OpenCV |
| 4 | `BRISK` | BRISK | BRISK | Hamming (Binary) | OpenCV |
| 5 | `FAST+BRIEF` | FAST | BRIEF | Hamming (Binary) | OpenCV |
| 6 | `KAZE` | KAZE | KAZE | L2 (Float) | OpenCV |
| 7 | `GFTT+BRIEF` | Good Features to Track | BRIEF | Hamming (Binary) | OpenCV |
| 8 | `GFTT+SIFT` | Good Features to Track | SIFT | L2 (Float) | OpenCV |
| 9 | `STAR+BRIEF` | Star/CenSurE | BRIEF | Hamming (Binary) | OpenCV |
| 10 | `ORB+FREAK` | ORB | FREAK | Hamming (Binary) | OpenCV |
| 11 | `FAST+FREAK` | FAST | FREAK | Hamming (Binary) | OpenCV |
| 12 | `SuperPoint` | SuperPoint (Learned) | SuperPoint (Learned) | L2 (Float) | PyTorch / Kornia |
| 13 | `ALIKED` | ALIKED (Learned) | ALIKED (Learned) | L2 (Float) | PyTorch / Kornia |
| 14 | `XFeat` | XFeat (Learned) | XFeat (Learned) | L2 (Float) | PyTorch / Kornia |

---

## 4. Mathematical Formulation

### 4.1 Observation Space ($34\text{ Dimensions}$)

The Level 2 observation vector $\mathbf{o}_t \in \mathbb{R}^{34}$ provides complete situational awareness:

#### A. Visual Pair Statistics ($9\text{ Dimensions}$)
Fast, pre-match statistical features extracted directly from images $(I_1, I_2)$ and normalized:
1. `mean_brightness`: Average grayscale intensity
2. `rms_contrast`: Root-mean-square contrast
3. `laplacian_sharpness`: Variance of the Laplacian (focus / blur)
4. `entropy`: Shannon entropy (information and texture richness)
5. `colorfulness`: Hasler-Süsstrunk colorfulness metric
6. `aspect_ratio`: Log ratio of image dimensions
7. `brightness_ratio`: Relative illumination change between images
8. `contrast_ratio`: Relative contrast difference
9. `sharpness_ratio`: Relative blur difference

#### B. Active Feature Preset ($14\text{ Dimensions}$)
A 14-dimensional one-hot vector indicating which of the 14 presets is currently active.

#### C. Dynamic Runtime Indicators ($11\text{ Dimensions}$)
1. Requested Budget: $N / 4000$
2. Effective Keypoints: $\min(N, \text{Pool Size}) / 4000$
3. Pool Ceiling: $\text{Pool Size} / 4000$
4. Accuracy Score: Current homography / pose accuracy
5. RANSAC Inlier Ratio: $N_{\text{inliers}} / N_{\text{matches}}$
6. Normalized Match Count: $\min(N_{\text{matches}}, 1000) / 1000$
7. Trajectory Step Fraction: $t / 15$
8. Delta Accuracy ($\Delta A$): $A_t - A_{t-1}$
9. Delta Utility ($\Delta \mathcal{U}$): $\mathcal{U}_t - \mathcal{U}_{t-1}$
10. Plateau Indicator: $1.0$ if effective keypoints reached detector pool ceiling, else $0.0$
11. Gap from Episode Best: $\mathcal{U}_t - \mathcal{U}_{\text{best}} \le 0$

---

### 4.2 Utility & Step Reward Formulation

- **State Utility:**
  $$\mathcal{U}_t = \text{accuracy}_t - \lambda \times \left(\frac{N_{\text{eff}}}{N_{\text{max}}}\right)$$
  where $N_{\text{max}} = 4000$ and $\lambda = 0.1$ is the cost weight.
- **Potential-Based Step Reward:**
  $$R_t = \begin{cases} 0, & \text{if action is } \text{STOP} \\ (\mathcal{U}_t - \mathcal{U}_{t-1}) - 0.01, & \text{otherwise} \end{cases}$$
  The small step penalty ($0.01$) drives the agent to reach peak utility quickly and issue `STOP`.

---

## 5. Curriculum Training Strategy (Reverse Curriculum)

Training is conducted in reverse hierarchy:
1. **Stage 1 (Level 2 Keypoint Agent):**
   - Trained for **150,000 - 170,000 steps** across randomized presets on HPatches.
   - Learns keypoint budgeting dynamics across different image pairs and algorithms.
   - Checkpoint saved to `checkpoints/level2_14preset_best/best_model.zip`.
2. **Stage 2 (Level 1 Pipeline Selection Agent):**
   - Level 2 is frozen as the downstream reward evaluator.
   - Level 1 is trained for **30,000 steps** with PPO configured as a single-step contextual bandit ($\gamma = 0$).
   - Learns to select the feature preset that maximizes the final downstream utility achieved by Level 2.
   - Checkpoint saved to `checkpoints/level1_best/best_model.zip`.

---

## 6. Multi-Dataset Evaluation & AUC Benchmarks

The full hierarchical system (Level 1 + Level 2) and Level 2 standalone policy were comprehensively evaluated across 3 major benchmarks in **Area Under the Curve (AUC)** terms:

### 1. HPatches Benchmark (Planar Homography)
- Evaluated on **580 pairs** across 116 sequences under illumination and viewpoint variations.
- Metric: **Homography AUC @ 1px, 3px, and 5px**.
- Level 2 budgeting significantly outperforms fixed-budget baselines across all 14 presets, with deep-learned presets (`ALIKED`, `SuperPoint`, `XFeat`) reaching peak performance with minimal keypoint overhead.

### 2. MegaDepth-1500 (Outdoor Multi-View 3D Reconstruction)
- Evaluated on the 225-pair validation split across wide baselines, depth discontinuities, and extreme viewpoint shifts.
- Metric: **Relative Camera Pose Error AUC @ 5°, 10°, and 20°** ($e_R = \arccos\frac{\text{Tr}(R^{\top}R_{\text{gt}})-1}{2}$, $e_t = \arccos\frac{t^{\top}t_{\text{gt}}}{\|t\|\|t_{\text{gt}}\|}$).
- Demonstrated zero-shot generalization of the HPatches-trained policies to unconstrained outdoor 3D scenes.

### 3. Aachen Day-Night v1.1 (Visual Localization Under Extreme Illumination)
- Evaluated on 225 pairs (150 daytime reference pairs + 75 challenging night query pairs).
- Metric: **Pose Recall @ (0.25m / 2°), (0.50m / 5°), (5.00m / 10°)** and **Pose AUC @ 5°, 10°, 20°**.
- Level 1 dynamically routes dark nighttime scenes to robust learned presets (`ALIKED` / `SuperPoint`), avoiding classical failure modes.

### Automated Presentation Deck
All evaluated AUC tables, per-preset breakdowns, and comparative plots are compiled into a comprehensive 16-slide PowerPoint deck:
```bash
python scripts/build_auc_results_presentation.py
# Generates: experiments/final_report/AFM_RL_AUC_Evaluation_Results.pptx
```

---

## 7. Interactive Live Demo / Inference on Custom Images

You can run arbitrary image pairs through the trained hierarchical pipeline using the interactive demo tool.

### 3 Ways to Provide Image Pairs:

#### Option A: Drag & Drop into `demo_input/`
Place any two images (e.g. `img1.jpg` and `img2.jpg`) into the [`demo_input/`](demo_input/) folder and run:
```bash
python scripts/demo_pair.py
```

#### Option B: Native Windows File Picker Dialog
Run with `--gui` to pick files via the Windows Explorer dialog:
```bash
python scripts/demo_pair.py --gui
```
*(Or simply double-click `run_demo.bat` in the root folder).*

#### Option C: Command-Line Paths
```bash
python scripts/demo_pair.py --img1 "path/to/img1.jpg" --img2 "path/to/img2.jpg"
```
*(Or test benchmark samples with `--sample hpatches` or `--sample megadepth`).*

### Demo Output:
- Prints Level 1 chosen preset (e.g., `ALIKED`).
- Prints step-by-step Level 2 budgeting trajectory (e.g., $500 \to 1000 \to 1500 \to \text{STOP}$).
- Displays and saves the matched inlier visualization to `demo_output/matched_result.png`.

---

## 8. Repository Structure

```text
afm-rl/
├── configs/
│   └── split.json                     # Stratified 70/15/15 HPatches train/val/test split
├── data/
│   ├── hpatches/                      # HPatches planar homography dataset
│   ├── megadepth/                     # MegaDepth-1500 outdoor dataset
│   ├── images_upright/                # Aachen Day-Night v1.1 images
│   └── pool_cache/                    # Precomputed candidate feature pools
├── demo_input/                        # Folder for dropping custom images for inference
├── demo_output/                       # Output visualizations (matched_result.png)
├── src/
│   ├── level1_env.py                  # Level 1 Gymnasium Contextual Bandit environment
│   ├── level2_env.py                  # Level 2 Gymnasium sequential PPO environment (34-dim)
│   ├── pipeline_utils.py              # 14 feature presets (OpenCV + Kornia SuperPoint/ALIKED/XFeat)
│   ├── image_stats.py                 # 9-dimensional visual pair feature extraction
│   ├── metrics.py                     # Homography accuracy, corner error, and reward computation
│   ├── hpatches.py                    # HPatches dataset loader and homography reader
│   ├── megadepth.py                   # MegaDepth loader and pose extraction
│   └── megadepth_env.py               # MegaDepth RL evaluation environment
├── scripts/
│   ├── demo_pair.py                   # Interactive live inference script (CLI + GUI)
│   ├── train_level1.py                # Level 1 Contextual Bandit training script
│   ├── train_level2.py                # Level 2 14-preset PPO training script
│   ├── evaluate_level1.py             # Evaluation of Level 1 preset selection
│   ├── evaluate_vs_baselines.py       # Full HPatches benchmark evaluation
│   ├── eval_14preset_test.py          # Level 2 test split evaluation across 14 presets
│   ├── eval_megadepth_hierarchical.py # MegaDepth zero-shot evaluation (L1 + L2)
│   ├── eval_aachen_hierarchical.py    # Aachen Day-Night evaluation (L1 + L2)
│   └── build_auc_results_presentation.py # 16-slide PowerPoint results deck generator
├── checkpoints/
│   ├── level1_best/best_model.zip     # Best trained Level 1 Pipeline Agent
│   ├── level2_14preset_best/best_model.zip # Best trained Level 2 14-Preset Agent
│   └── level2_deltaobs_best/          # Level 2 checkpoint with delta observations
├── run_demo.bat                       # One-click Windows demo launcher
└── README.md                          # Project documentation
```

---

## 9. Quickstart: Reproduction & Evaluation Commands

### 1. Run the Interactive Demo
```bash
python scripts/demo_pair.py --gui
```

### 2. Reproduce Hierarchical MegaDepth Evaluation
```bash
python scripts/eval_megadepth_hierarchical.py
```

### 3. Reproduce Hierarchical Aachen Day-Night Evaluation
```bash
python scripts/eval_aachen_hierarchical.py
```

### 4. Generate the Results Presentation Deck
```bash
python scripts/build_auc_results_presentation.py
```
Outputs the presentation file to `experiments/final_report/AFM_RL_AUC_Evaluation_Results.pptx`.
