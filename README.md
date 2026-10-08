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
| **Decision Type** | Single-shot contextual decision | Sequential multi-step decision (Max 15 steps) |
| **Algorithm** | Contextual Bandit | **PPO (Proximal Policy Optimization)** |
| **Action Space** | 11 Presets + Matchers + Thresholds | Discrete(4): `+500`, `+1000`, `-500`, `STOP` |
| **Reward Signal** | Final score Level 2 achieves | Potential-based utility delta ($\Delta \text{Utility} - \text{step\_penalty}$) |
| **Status** | Next stage to train (using frozen Level 2) | **Trained and Frozen** |

---

## 3. Curriculum Training Strategy & Current Status

While decision flow at inference is **Level 1 $\to$ Level 2**, the training curriculum is executed in reverse:

1. **Stage 1 (Completed & Frozen):** Train the **Level 2 Keypoint Agent** across randomized feature presets until it becomes competent at budgeting keypoints.
   - Level 2 is fully trained and frozen.
   - The reported presentation experiment was trained for approximately **170,000 total steps** using PPO with Stable-Baselines3 on the HPatches training split.
   - Best checkpoint selection was guided by validation performance via `EvalCallback`.
2. **Stage 2 (Next Step):** Freeze Level 2 and train the **Level 1 Pipeline Agent** using the score achieved by the frozen Level 2 agent as its reward signal.
3. **Stage 3 (Optional):** Joint fine-tuning pass.

*Why reverse training?* Level 1 cannot fairly evaluate whether a pipeline like AKAZE or SIFT is effective unless the attached keypoint-tuning agent is already competent at finding its optimal budget.

---

## 4. Mathematical Formulation

### 4.1 Observation Space ($31\text{ Dimensions}$)

The observation vector $\mathbf{o}_t \in \mathbb{R}^{31}$ provides the agent with complete situational awareness across three categories:

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
A one-hot vector indicating which of the 11 CV pipelines is currently active:
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

#### C. Runtime & Delta State Indicators ($11\text{ Dimensions}$)
Dynamic metrics tracking the current step, matching quality, and trajectory progress:
1. **Requested Keypoint Budget:** $\frac{N_{\text{req}}}{N_{\text{max}}} = \frac{N}{4000}$
2. **Effective Keypoint Count:** $\frac{N_{\text{eff}}}{N_{\text{max}}} = \frac{\min(N, \text{Pool Size})}{4000}$ *(charges real keypoints used)*
3. **Detector Pool Ceiling:** $\frac{\text{Pool Size}}{N_{\text{max}}} = \frac{\min(|P_1|, |P_2|)}{4000}$ *(maximum keypoints detector can find)*
4. **Current Accuracy Score:** $A_t = \exp\left(-\frac{\text{Corner Error}}{5.0}\right) \in (0, 1]$
5. **Last RANSAC Inlier Ratio:** $\frac{N_{\text{inliers}}}{N_{\text{matches}}} \in [0, 1]$
6. **Normalized Match Count:** $\frac{\min(N_{\text{matches}}, 1000)}{1000} \in [0, 1]$
7. **Episode Step Fraction:** $\frac{t}{T_{\text{max}}} = \frac{t}{15}$
8. **Delta Accuracy ($\Delta A$):** $A_t - A_{t-1}$ *(accuracy change from the previous step)*
9. **Delta Utility ($\Delta \mathcal{U}$):** $\mathcal{U}_t - \mathcal{U}_{t-1}$ *(utility change from the previous step)*
10. **Plateau Indicator:** $1.0$ if $N_{\text{eff}} \ge \text{Pool Size}$, else $0.0$ *(flags detector pool saturation)*
11. **Gap / Distance from Best Utility:** $\mathcal{U}_t - \mathcal{U}_{\text{best}} \le 0$ *(distance below the best utility seen in this episode)*

---

### 4.2 Reward & Utility Formulation

The objective is to maximize homography accuracy while penalizing computational overhead and unnecessary exploration steps.

#### 1. Accuracy Metric & Homography Success
Homography estimation quality is measured by projecting the 4 image corners using the ground-truth matrix $H_{\text{gt}}$ versus the RANSAC-estimated matrix $H_{\text{est}}$:
$$\text{Corner Error} = \frac{1}{4} \sum_{i=1}^{4} \|\operatorname{proj}(H_{\text{gt}}, \mathbf{c}_i) - \operatorname{proj}(H_{\text{est}}, \mathbf{c}_i)\|_2$$

- **Homography Success ($h_{\text{ok}}$):** A match is considered successful ($h_{\text{ok}} = 1$) when $\text{Corner Error} < 3.0\text{ pixels}$.
- **Smooth Reward Accuracy:**
$$\text{reward\_acc} = \begin{cases} \exp\left(-\dfrac{\text{Corner Error}}{5.0}\right), & \text{if valid } H_{\text{est}} \text{ exists} \\ 0, & \text{if RANSAC fails / } \text{Corner Error} = \infty \end{cases}$$

#### 2. State Utility
Each state has an objective utility value balancing accuracy and keypoint cost:
$$\mathcal{U}_t = \text{reward\_acc}_t - \lambda \times \left(\frac{N_{\text{eff}}}{N_{\text{max}}}\right)$$
* $N_{\text{max}} = 4000$ keypoints.
* $\lambda = 0.1$: Cost-penalty weight (ensures the agent only requests more keypoints if the accuracy gain exceeds the computational price).

#### 3. Potential-Based Step Reward
To provide immediate dense feedback without distorting optimal policy convergence, the per-step reward is computed as the utility delta minus a small step penalty:
$$R_t = \begin{cases} 0, & \text{if action is } \text{STOP} \\ (\mathcal{U}_t - \mathcal{U}_{t-1}) - \text{step\_penalty}, & \text{otherwise} \end{cases}$$
* $\text{step\_penalty} = 0.01$: Exploration penalty ensuring the agent learns when increasing/decreasing the keypoint budget is helpful and learns to terminate via `STOP` once peak utility is reached.
* Maximum episode length: $T_{\text{max}} = 15$ steps.

---

## 5. Dataset & Benchmark Results

### 5.1 Dataset Partitioning
The benchmark uses the **HPatches** dataset consisting of **116 sequences** and approximately **580 image pairs** (covering illumination and viewpoint changes), partitioned into:
- **70% Train** (for training the Level 2 RL policy)
- **15% Validation** (for model selection and `EvalCallback` checkpoints)
- **15% Test** (for final held-out evaluation)

### 5.2 Key Level 2 Findings (Reported Experiment)
- **All 11 Presets Improved:** Adaptive keypoint selection using the trained/frozen Level 2 agent improves homography success rates across all 11 CV pipeline presets compared to fixed-budget baselines.
- **Highest Gain:** The largest improvement was observed on **`STAR+BRIEF`**, which achieved an increase of approximately **+38 percentage points** in success rate.
- **Next Milestone:** Use the frozen Level 2 policy as the downstream evaluator to train the Level 1 Pipeline Selection Agent.

---

## 6. Repository Structure

```text
afm-rl/
├── configs/
│   └── split.json            # Stratified 70/15/15 HPatches train/val/test split
├── data/
│   ├── hpatches/             # Benchmark images and ground truth homographies
│   └── pool_cache/           # Precomputed keypoint pool cache (.npz)
├── src/
│   ├── level2_env.py         # Gymnasium environment for Level 2 (31-dim observation, Discrete(4))
│   ├── pipeline_utils.py     # OpenCV feature detection, matching, and disk caching
│   ├── image_stats.py        # 9-dimensional image feature extraction
│   ├── metrics.py            # Corner error and homography reward evaluation
│   ├── hpatches.py           # Dataset indexing and pair loading
│   └── cv_compat.py          # OpenCV namespace compatibility layer
├── scripts/
│   ├── verify_hpatches.py    # Dataset integrity verification
│   ├── precompute_pools.py   # One-time feature pool generation
│   ├── train_level2.py       # Level 2 PPO training script (150k timesteps)
│   ├── baseline_always_stop.py # Default-N baseline comparison
│   ├── check_action_usage.py # Action distribution inspector
│   ├── inspect_level2_policy.py# Step-by-step rollout inspector
│   └── evaluate_vs_baselines.py# Full benchmark evaluation vs fixed baselines
├── checkpoints/              # Trained PPO model weights (e.g., level2_deltaobs_best)
└── logs/                     # TensorBoard logs (level2_tb_deltaobs) and eval records
```

---

## 7. Installation & Reproducing Level 2 Training

### 1. Install Dependencies
```bash
pip install -r requirements.txt
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
The default training script (`scripts/train_level2.py`) trains a 31-dimensional PPO agent with the following hyperparameters:
* `TOTAL_TIMESTEPS = 150_000` *(Note: The presentation reports ~170,000 total steps across its training runs)*
* `n_steps = 1024`
* `batch_size = 128`
* `learning_rate = 3e-4`
* `ent_coef = 0.03`
* `gamma = 0.99`

```bash
python scripts/train_level2.py
```
Checkpoints are saved to `checkpoints/level2_deltaobs` and `checkpoints/level2_deltaobs_best`.

### 5. Real-Time Dashboard (TensorBoard)
Monitor training rewards and episode metrics in your browser:
```bash
tensorboard --logdir logs/level2_tb_deltaobs
```
Open `http://localhost:6006` to view live training graphs.

### 6. Evaluate Policy
```bash
# Evaluate trained model against fixed baselines across all 11 presets
python scripts/evaluate_vs_baselines.py

# Inspect action distribution (+500, +1000, -500, STOP)
python scripts/check_action_usage.py

# Run detailed rollouts on validation pairs
python scripts/inspect_level2_policy.py
```

---

## 8. Hyperparameter Tuning Guide

| Scenario | Root Cause | Parameter to Adjust |
| :--- | :--- | :--- |
| **Agent always chooses `STOP` immediately** | Cost penalty is too high | Decrease `DEFAULT_LAMBDA` (`0.1` $\to$ `0.05`) in `src/level2_env.py` |
| **Agent always maxes out keypoints ($N=4000$)** | Cost penalty is too low | Increase `DEFAULT_LAMBDA` (`0.1` $\to$ `0.2`) in `src/level2_env.py` |
| **Policy collapses into a single action** | Insufficient exploration | Increase `ent_coef` (`0.03` $\to$ `0.06`) in `scripts/train_level2.py` |
| **Noisy / unconverged reward curve** | Needs more experience | Increase `TOTAL_TIMESTEPS` (`150_000` $\to$ `200_000+`) in `scripts/train_level2.py` |

