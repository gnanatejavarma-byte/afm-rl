# AFM-RL: Adaptive Feature Matching Using Hierarchical Reinforcement Learning

An intelligent, two-level hierarchical reinforcement learning system that dynamically selects both the optimal feature matching pipeline and the minimal keypoint budget per image pair to optimize homography accuracy vs. computational cost.

---

## 1. Problem Statement
Traditional computer vision pipelines (such as SIFT, ORB, and AKAZE) use a fixed keypoint budget and static algorithm configurations for all images:
- **Easy, highly-textured scenes** waste computation and energy by extracting thousands of unnecessary keypoints.
- **Challenging, blurry, or low-texture scenes** underperform or fail completely because they need a stronger pipeline or higher keypoint density.

**AFM-RL** replaces static configurations with an adaptive RL decision system that inspects scene properties on the fly and tunes the matching process.

---

## 2. Two-Level Hierarchical Architecture

The system splits the decision into two distinct levels matching human intuition: *first choose the tool, then decide how much of it to use*.

```
                      Image Pair
                          │
                          ▼
            Extract Visual Characteristics 
         (Entropy, Blur, Corner Density, Gradients)
                          │
                          ▼
             LEVEL 1: Pipeline Agent (Contextual Bandit)
             └── Selects: Feature Preset + Matcher + Ratio Threshold
                          │
                          ▼
             LEVEL 2: Keypoint Agent (PPO / Stable-Baselines3)
             └── Dynamically tunes keypoint budget: [+500, +1000, -500, STOP]
                          │
                          ▼
             Final Matches + High Homography Accuracy / Minimal Cost
```

### Hierarchy Comparison

| Feature | Level 1: Pipeline Agent | Level 2: Keypoint Agent |
| :--- | :--- | :--- |
| **Role** | Chooses algorithm recipe | Tunes keypoint count ($N$) |
| **Decision Type** | Single-shot contextual decision | Sequential multi-step decision |
| **Algorithm** | Contextual Bandit | **PPO (Proximal Policy Optimization)** |
| **Action Space** | 11 Presets + Matchers + Thresholds | Discrete: `+500`, `+1000`, `-500`, `STOP` |
| **Reward Signal** | Final score Level 2 achieves | Potential-based utility delta ($\Delta \text{Utility} - \text{tax}$) |

---

## 3. Curriculum Training Strategy

While decision flow at inference is **Level 1 $\to$ Level 2**, the training curriculum is executed in reverse:

1. **Stage 1 (Current Stage):** Train the **Level 2 Keypoint Agent** across randomized feature presets until it becomes competent at budgeting keypoints.
2. **Stage 2:** Freeze Level 2 and train the **Level 1 Pipeline Agent** using the score achieved by the frozen Level 2 agent as its reward.
3. **Stage 3 (Optional):** Joint fine-tuning pass.

*Why reverse training?* Level 1 cannot fairly evaluate whether a pipeline like AKAZE or SIFT is effective unless the attached keypoint-tuning agent is already competent at finding its optimal budget.

---

## 4. Mathematical Formulation

### 4.1 Observation Space ($27\text{ Dimensions}$)

The observation vector $\mathbf{o}_t \in \mathbb{R}^{27}$ provides the agent with complete situational awareness across three categories:

#### A. Visual Pair Statistics ($9\text{ Dimensions}$)
Fast, pre-match statistical features extracted directly from grayscale images $(I_1, I_2)$ and normalized:
1. **Average Texture Richness:** Mean Shannon entropy $\frac{1}{2}(e_1 + e_2)$
2. **Texture Discrepancy:** Absolute entropy difference $|e_1 - e_2|$
3. **Average Blur:** Mean Laplacian variance $\frac{1}{2}\big(\log(1+b_1) + \log(1+b_2)\big)$
4. **Blur Discrepancy:** Difference in blur levels $|\log(1+b_1) - \log(1+b_2)|$
5. **Average Corner Density:** Mean Shi-Tomasi/Harris corner density $\frac{1}{2}(c_1 + c_2)$
6. **Corner Density Discrepancy:** Absolute difference in corner density $|c_1 - c_2|$
7. **Average Gradient Magnitude:** Mean Sobel edge strength $\frac{1}{2}(\mu_{g1} + \mu_{g2})$
8. **Gradient Spread:** Mean Sobel gradient standard deviation $\frac{1}{2}(\sigma_{g1} + \sigma_{g2})$
9. **Mean Intensity Shift:** Absolute difference in illumination $|\bar{I}_1 - \bar{I}_2|$

#### B. Active Feature Preset ($11\text{ Dimensions}$)
A one-hot vector indicating which CV pipeline is currently active:
- `ORB` (ORB detector + ORB descriptor, Binary)
- `SIFT` (SIFT detector + SIFT descriptor, Float)
- `AKAZE` (AKAZE detector + AKAZE descriptor, Binary)
- `BRISK` (BRISK detector + BRISK descriptor, Binary)
- `FAST+BRIEF` (FAST detector + BRIEF descriptor, Binary)
- `KAZE` (KAZE detector + KAZE descriptor, Float)
- `GFTT+BRIEF` (Good Features To Track + BRIEF descriptor, Binary)
- `GFTT+SIFT` (Good Features To Track + SIFT descriptor, Float)
- `STAR+BRIEF` (Star/CenSurE detector + BRIEF descriptor, Binary)
- `ORB+FREAK` (ORB detector + FREAK descriptor, Binary)
- `FAST+FREAK` (FAST detector + FREAK descriptor, Binary)

#### C. Runtime State Indicators ($7\text{ Dimensions}$)
Dynamic metrics tracking the current step and matching quality:
1. **Requested Keypoint Budget:** $\frac{N_{\text{req}}}{N_{\text{max}}} = \frac{N}{4000}$
2. **Effective Keypoint Count:** $\frac{N_{\text{eff}}}{N_{\text{max}}} = \frac{\min(N, \text{Pool Size})}{4000}$ *(charges real keypoints used)*
3. **Detector Pool Ceiling:** $\frac{\text{Pool Size}}{N_{\text{max}}} = \frac{\min(|P_1|, |P_2|)}{4000}$ *(maximum keypoints detector can find)*
4. **Last Accuracy Score:** $A_t = \exp\left(-\frac{\text{Corner Error}}{5.0}\right) \in (0, 1]$
5. **Last RANSAC Inlier Ratio:** $\frac{N_{\text{inliers}}}{N_{\text{matches}}} \in [0, 1]$
6. **Normalized Match Count:** $\frac{\min(N_{\text{matches}}, 1000)}{1000} \in [0, 1]$
7. **Episode Step Fraction:** $\frac{t}{T_{\text{max}}} = \frac{t}{15}$

---

### 4.2 Reward & Utility Formulation

The objective is to maximize homography accuracy while penalizing computational overhead and unnecessary exploration steps.

#### 1. Accuracy Metric
Homography estimation quality is measured by projecting the 4 image corners using the ground-truth matrix $H_{\text{gt}}$ versus the RANSAC-estimated matrix $H_{\text{est}}$:
$$\text{Corner Error} = \frac{1}{4} \sum_{i=1}^{4} \|\operatorname{proj}(H_{\text{gt}}, \mathbf{c}_i) - \operatorname{proj}(H_{\text{est}}, \mathbf{c}_i)\|_2$$

The smooth accuracy score is defined as:
$$\text{Accuracy} = \begin{cases} \exp\left(-\dfrac{\text{Corner Error}}{5.0}\right), & \text{if valid } H_{\text{est}} \text{ exists} \\ 0, & \text{if RANSAC fails / } \text{Corner Error} = \infty \end{cases}$$

#### 2. State Utility
Each state has an objective utility value balancing accuracy and keypoint cost:
$$\mathcal{U}_t = \text{Accuracy}_t - \lambda \times \left(\frac{N_{\text{eff}}}{4000}\right)$$
* $\lambda = 0.1$: Cost-penalty weight (ensures an agent only requests more keypoints if the accuracy gain exceeds the computational price).

#### 3. Potential-Based Step Reward
To provide immediate dense feedback without distorting optimal policy convergence, the per-step reward is computed as the utility delta minus a small exploration tax:
$$R_t = \begin{cases} 0, & \text{if action is } \text{STOP} \\ (\mathcal{U}_t - \mathcal{U}_{t-1}) - \eta, & \text{otherwise} \end{cases}$$
* $\eta = 0.01$: Step penalty (exploration tax) ensuring the agent terminates via `STOP` as soon as peak utility is reached.

---

## 5. Repository Structure

```text
afm-rl/
├── configs/
│   └── split.json            # Stratified 70/15/15 HPatches train/val/test split
├── data/
│   ├── hpatches/             # Benchmark images and ground truth homographies
│   └── pool_cache/           # Precomputed keypoint pool cache (.npz)
├── src/
│   ├── level2_env.py         # Gymnasium environment for Level 2 Keypoint Agent
│   ├── pipeline_utils.py     # OpenCV feature detection, matching, and disk caching
│   ├── image_stats.py        # 9-dimensional image feature extraction
│   ├── metrics.py            # Corner error and homography reward evaluation
│   ├── hpatches.py           # Dataset indexing and pair loading
│   └── cv_compat.py          # OpenCV namespace compatibility layer
├── scripts/
│   ├── verify_hpatches.py    # Dataset integrity verification
│   ├── precompute_pools.py   # One-time feature pool generation
│   ├── train_level2.py       # Level 2 PPO training script
│   ├── baseline_always_stop.py # Default-N baseline comparison
│   ├── check_action_usage.py # Action distribution inspector
│   └── inspect_level2_policy.py# Step-by-step rollout inspector
├── checkpoints/              # Trained PPO model weights (.zip)
└── logs/                     # TensorBoard training curves and evaluation histories
```

---

## 6. Installation & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install tensorboard
```

### 2. Verify Dataset
Ensure the HPatches dataset is extracted into `data/hpatches/`:
```bash
python scripts/verify_hpatches.py
```

### 3. Precompute Keypoint Pools (Run Once)
Precomputes candidate keypoint pools for fast RL simulation:
```bash
python scripts/precompute_pools.py
```

### 4. Train Level 2 Keypoint Agent
```bash
python scripts/train_level2.py
```

### 5. Real-Time Dashboard (TensorBoard)
Monitor training rewards and episode lengths in your browser:
```bash
tensorboard --logdir logs/level2_tb
```
Open `http://localhost:6006` to view live training graphs.

### 6. Evaluate Policy
```bash
# Compare trained model against naive baseline (N=500 always)
python scripts/baseline_always_stop.py

# Inspect action distribution (+500, +1000, -500, STOP)
python scripts/check_action_usage.py

# Run detailed rollouts on validation pairs
python scripts/inspect_level2_policy.py
```

---

## 7. Hyperparameter Tuning Guide

| Scenario | Root Cause | Parameter to Adjust |
| :--- | :--- | :--- |
| **Agent always chooses `STOP` immediately** | Cost penalty is too high | Decrease `DEFAULT_LAMBDA` (`0.1` $\to$ `0.05`) in `src/level2_env.py` |
| **Agent always maxes out keypoints ($N=4000$)** | Cost penalty is too low | Increase `DEFAULT_LAMBDA` (`0.1` $\to$ `0.2`) in `src/level2_env.py` |
| **Policy collapses into a single action** | Insufficient exploration | Increase `ent_coef` (`0.03` $\to$ `0.06`) in `scripts/train_level2.py` |
| **Noisy / unconverged reward curve** | Needs more experience | Increase `TOTAL_TIMESTEPS` (`50_000` $\to$ `100_000`) in `scripts/train_level2.py` |
