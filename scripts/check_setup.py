import cv2, numpy as np, gymnasium, stable_baselines3, torch

print("OpenCV", cv2.__version__, "| gymnasium", gymnasium.__version__,
      "| SB3", stable_baselines3.__version__, "| torch", torch.__version__)

checks = {
    "ORB":    lambda: cv2.ORB_create(),
    "SIFT":   lambda: cv2.SIFT_create(),
    "AKAZE":  lambda: cv2.AKAZE_create(),
    "BRISK":  lambda: cv2.BRISK_create(),
    "KAZE":   lambda: cv2.KAZE_create(),
    "FAST":   lambda: cv2.FastFeatureDetector_create(),
    "GFTT":   lambda: cv2.GFTTDetector_create(),
    "STAR":   lambda: cv2.xfeatures2d.StarDetector_create(),
    "BRIEF":  lambda: cv2.xfeatures2d.BriefDescriptorExtractor_create(),
    "FREAK":  lambda: cv2.xfeatures2d.FREAK_create(),
    "BFMatcher": lambda: cv2.BFMatcher(cv2.NORM_HAMMING),
    "FLANN":  lambda: cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {}),
}
for name, make in checks.items():
    try:
        make(); print(f"OK    {name}")
    except Exception as e:
        print(f"FAIL  {name}: {e}")

# tiny end-to-end SB3 smoke test
import gymnasium as gym
from stable_baselines3 import PPO
PPO("MlpPolicy", gym.make("CartPole-v1"), verbose=0).learn(2000)
print("SB3 PPO smoke test passed")