import cv2
import numpy as np

N_STATS = 9

def _entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def _blur(img):
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def _corner_density(img):
    corners = cv2.goodFeaturesToTrack(img, maxCorners=2000, qualityLevel=0.01, minDistance=3)
    n = 0 if corners is None else len(corners)
    return n / (img.shape[0] * img.shape[1])

def _gradient_stats(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag.mean()), float(mag.std())

def compute_pair_stats(img1, img2):
    """Raw, unnormalized stats. Cache these per (seq, k) since they never change."""
    e1, e2 = _entropy(img1), _entropy(img2)
    b1, b2 = _blur(img1), _blur(img2)
    c1, c2 = _corner_density(img1), _corner_density(img2)
    g1m, g1s = _gradient_stats(img1)
    g2m, g2s = _gradient_stats(img2)
    intensity_diff = float(abs(img1.mean() - img2.mean()))
    return np.array([
        (e1 + e2) / 2,        # 1. avg texture richness
        abs(e1 - e2),         # 2. texture-richness change between images
        (b1 + b2) / 2,        # 3. avg blur (Laplacian variance)
        abs(b1 - b2),         # 4. blur change between images
        (c1 + c2) / 2,        # 5. avg corner density
        abs(c1 - c2),         # 6. corner-density change
        (g1m + g2m) / 2,      # 7. avg gradient magnitude
        (g1s + g2s) / 2,      # 8. avg gradient magnitude spread
        intensity_diff,       # 9. mean-intensity shift (illumination signal)
    ], dtype=np.float32)

_SCALE = np.array([8.0, 4.0, 1.0, 1.0, 0.05, 0.05, 60.0, 40.0, 60.0], dtype=np.float32)

def normalize_stats(raw):
    x = raw.copy()
    x[2] = np.log1p(x[2])   # avg blur
    x[3] = np.log1p(x[3])   # blur change
    x = x / _SCALE
    return np.clip(x, -5.0, 5.0).astype(np.float32)