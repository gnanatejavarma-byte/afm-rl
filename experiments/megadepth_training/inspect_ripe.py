import PyPDF2
import os

pdf_path = "RIPE_paper.pdf"
reader = PyPDF2.PdfReader(pdf_path)
print(f"Total pages: {len(reader.pages)}")

keywords = ["megadepth", "auc", "pose", "hpatches", "corner", "threshold", "sampson", "essential", "angular"]

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    lower = text.lower()
    matches = [k for k in keywords if k in lower]
    if "megadepth" in lower or "auc" in lower or "hpatches" in lower:
        print(f"\n================ PAGE {i+1} ================")
        lines = text.split("\n")
        for line in lines:
            line_str = line.encode("ascii", "replace").decode("ascii")
            if any(k in line.lower() for k in ["auc", "megadepth", "hpatches", "metric", "pose error", "5", "10", "20", "3px", "5px", "corner", "rotation", "translation", "trapezoid", "cumulative", "evaluation"]):
                print("  ", line_str)
