import os
import glob
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from stable_baselines3 import PPO

print("=== 1. CHECKPOINT INSPECTION ===")
for root, dirs, files in os.walk("checkpoints"):
    for f in files:
        if f.endswith(".zip"):
            p = os.path.join(root, f)
            try:
                m = PPO.load(p)
                print(f"{p}: num_timesteps = {m.num_timesteps}")
            except Exception as e:
                print(f"{p}: error loading: {e}")

print("\n=== 2. EVALUATION LOGS INSPECTION ===")
for root, dirs, files in os.walk("logs"):
    for f in files:
        if f.endswith(".npz"):
            p = os.path.join(root, f)
            data = np.load(p)
            print(f"{p}: keys = {list(data.keys())}")
            if "timesteps" in data:
                print(f"  timesteps: {data['timesteps']}")
            if "results" in data:
                print(f"  results mean per eval: {data['results'].mean(axis=1)}")

print("\n=== 3. PPTX INSPECTION ===")
pptx_path = "Adaptive_Feature_Matching_Level2_Presentation.pptx"
if os.path.exists(pptx_path):
    with zipfile.ZipFile(pptx_path) as z:
        slides = [f for f in z.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")]
        slides.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
        for idx, s in enumerate(slides, 1):
            tree = ET.fromstring(z.read(s))
            texts = [node.text for node in tree.iter() if node.text]
            full_text = " ".join(texts)
            print(f"\n--- Slide {idx} ---")
            print(full_text)
