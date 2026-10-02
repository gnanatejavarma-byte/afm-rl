import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cv_compat import make

names = ["ORB", "SIFT", "AKAZE", "BRISK", "KAZE", "FastFeatureDetector",
         "GFTTDetector", "StarDetector", "BriefDescriptorExtractor", "FREAK"]
for name in names:
    try:
        obj = make(name)
        print(f"OK    {name} -> {type(obj).__name__}")
    except Exception as e:
        print(f"FAIL  {name}: {e}")