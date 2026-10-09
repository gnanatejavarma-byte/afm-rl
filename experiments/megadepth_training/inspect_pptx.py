import zipfile
import xml.etree.ElementTree as ET
import os

pptx_path = "Adaptive_Feature_Matching_Level2_Presentation.pptx"
with zipfile.ZipFile(pptx_path) as z:
    slides = [f for f in z.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")]
    slides.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    for idx, s in enumerate(slides, 1):
        tree = ET.fromstring(z.read(s))
        texts = [node.text for node in tree.iter() if node.text]
        full_text = " ".join(texts)
        # safe ascii print
        safe_text = full_text.encode("ascii", "replace").decode("ascii")
        print(f"\n--- Slide {idx} ---")
        print(safe_text)
