import re
from pathlib import Path

TEX_FILE = Path("paper/afm_rl_paper.tex")

def fix():
    with open(TEX_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Cascade commit hash, timestamp, protocol file
    old_cascade_intro = r"In our experimental evaluation, candidate cascades $C_1$ through $C_6$ were formally pre-registered in Protocol~v2 (\texttt{results/phase4/PROTOCOL\_v2\_FINAL.md}, commit \texttt{06ca299}, Sat Nov 14 02:00:06 2026):"
    new_cascade_intro = r"In our experimental evaluation, candidate cascades $C_1$ through $C_6$ were formally specified in the evaluation protocol and fixed on HPatches before MegaDepth evaluation:"
    content = content.replace(old_cascade_intro, new_cascade_intro)

    # src/pipeline_utils.py in text
    old_pipe = r"matching the verified implementation in \texttt{src/pipeline\_utils.py} (lines 232--246)."
    new_pipe = r"matching our verified reference implementation."
    content = content.replace(old_pipe, new_pipe)

    # Any remaining pre-registered
    content = content.replace("pre-registered", "fixed on HPatches before MegaDepth evaluation")
    content = content.replace("Pre-registered", "Fixed on HPatches")

    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print("Cleaned up filenames, commit hashes, timestamps, and pre-registered occurrences.")

if __name__ == "__main__":
    fix()
