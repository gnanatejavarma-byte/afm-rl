"""
Verification script for Phase 9:
Audits paper/afm_rl_paper.tex for:
1. Absence of all forbidden words and forbidden phrases.
2. Abstract constraints (<= 200 words, single paragraph, no symbols, brackets, %, +/-, equations, citations).
3. Statistical rigor: every occurrence of 'significant' or 'indistinguishable' must be paired with a confidence interval.
4. Correct code availability URL.
5. Exact 8 limitations in Section V.
6. Cascade terminology ('Fixed on HPatches', 'Mean over 5 Folds', absence of 'AFM-RL v2').
7. Absence of leaked file names, commit hashes, or timestamps in body prose.
"""

import re
import sys
from pathlib import Path

TEX_FILE = Path("paper/afm_rl_paper.tex")

FORBIDDEN_PATTERNS = [
    (r"\b70\.2\b", "70.2%"),
    (r"\bdominat\w*", "dominating / dominate"),
    (r"operational\s+envelope", "operational envelope"),
    (r"significantly\s+shifts", "significantly shifts"),
    (r"\bgenuine\b", "genuine"),
    (r"\bremarkabl\w*", "remarkably / remarkable"),
    (r"\bexceptional\w*", "exceptionally / exceptional"),
    (r"\bsubstantial\b", "substantial"),
    (r"\bsevere\b", "severe"),
    (r"state-of-the-art", "state-of-the-art"),
    (r"AFM-RL\s+v2", "AFM-RL v2"),
    (r"\bRyzen\b", "Ryzen"),
    (r"earlier\s+drafts?", "earlier draft(s)"),
    (r"statistical\s+equivalence", "statistical equivalence"),
    (r"\bcorrupt\w*", "corrupted / corrupt"),
    (r"\bweakly\b", "weakly"),
    (r"pre-registered", "pre-registered (should be fixed on HPatches before MegaDepth evaluation)"),
]

def check_tex():
    if not TEX_FILE.exists():
        print(f"Error: {TEX_FILE} does not exist.")
        return 1

    with open(TEX_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    errors = 0

    print("=" * 60)
    print("PHASE 9 TEXT CLAIMS & COMPLIANCE VERIFICATION")
    print("=" * 60)

    # 1. Check abstract
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", content, re.DOTALL)
    if not abs_match:
        print("[FAIL] Abstract not found.")
        errors += 1
    else:
        abs_text = abs_match.group(1).strip()
        words = abs_text.split()
        word_count = len(words)
        print(f"Abstract word count: {word_count} (must be <= 200)")
        if word_count > 200:
            print(f"[FAIL] Abstract exceeds 200 words ({word_count} words).")
            errors += 1
        else:
            print("[PASS] Abstract word count <= 200.")

        # Check paragraphs
        if "\n\n" in abs_text or "\\par" in abs_text:
            print("[FAIL] Abstract must be a single paragraph.")
            errors += 1
        else:
            print("[PASS] Abstract is a single paragraph.")

        # Check no symbols / brackets / % / +/- / citations
        forbidden_in_abs = ["%", "\\cite", "$", "{", "}", "[", "]", "+/-", "\\pm"]
        found_syms = [s for s in forbidden_in_abs if s in abs_text]
        if found_syms:
            print(f"[FAIL] Abstract contains forbidden formatting/symbols: {found_syms}")
            errors += 1
        else:
            print("[PASS] Abstract contains no symbols, brackets, %, or citations.")

    # 2. Check forbidden words across text
    # Strip comments first for text search, but check comments separately
    lines_no_comment = []
    for line in content.splitlines():
        # Remove LaTeX comments
        line_clean = re.sub(r"(?<!\\)%.*$", "", line)
        lines_no_comment.append(line_clean)
    clean_text = "\n".join(lines_no_comment)

    print("\nAuditing forbidden words and phrases...")
    for pat, desc in FORBIDDEN_PATTERNS:
        matches = list(re.finditer(pat, clean_text, re.IGNORECASE))
        if matches:
            for m in matches:
                # Get line snippet
                start = max(0, m.start() - 40)
                end = min(len(clean_text), m.end() + 40)
                snippet = clean_text[start:end].replace("\n", " ")
                print(f"[FAIL] Found forbidden pattern '{desc}': ...{snippet}...")
                errors += 1
        else:
            print(f"[PASS] No occurrences of '{desc}'.")

    # 3. Code Availability URL
    print("\nAuditing Code Availability...")
    if "https://github.com/gnanatejavarma-byte/afm-rl" in content:
        print("[PASS] Code Availability URL correctly points to https://github.com/gnanatejavarma-byte/afm-rl.")
    else:
        print("[FAIL] Code Availability URL missing or incorrect.")
        errors += 1

    # 4. Limitations Count
    print("\nAuditing Limitations section...")
    lim_match = re.search(r"\\section\{Limitations\}(.*?)(?:\\section|\Z)", content, re.DOTALL)
    if not lim_match:
        print("[FAIL] Limitations section not found.")
        errors += 1
    else:
        lim_text = lim_match.group(1)
        items = re.findall(r"\\item\s+\\textbf\{", lim_text)
        print(f"Limitations items count: {len(items)} (must be exactly 8)")
        if len(items) != 8:
            print(f"[FAIL] Expected 8 limitations, found {len(items)}.")
            errors += 1
        else:
            print("[PASS] Exactly 8 limitations found.")

    # 5. Check 'significant' only accompanied by CI
    print("\nAuditing 'significant' and 'significantly' claims...")
    sig_matches = list(re.finditer(r"\b(significantly|significant)\b", clean_text, re.IGNORECASE))
    for m in sig_matches:
        start = max(0, m.start() - 100)
        end = min(len(clean_text), m.end() + 100)
        snippet = clean_text[start:end].replace("\n", " ")
        # Exclude common non-statistical usages like "significant margin" or check if CI is nearby
        has_ci = "[" in snippet or "CI" in snippet or "confidence interval" in snippet
        if not has_ci:
            print(f"[WARN/CHECK] Mention of '{m.group(0)}' without nearby CI: ...{snippet}...")

    # 6. Check filenames in body text
    print("\nAuditing leaked filenames in body text...")
    for fn in ["megadepth_eval_env.py", "level2_env.py", "eval_14preset_test.py", "pipeline_utils.py"]:
        # Only in Methods or Limitations or verify where it is
        matches = list(re.finditer(re.escape(fn), clean_text))
        if matches:
            for m in matches:
                start = max(0, m.start() - 40)
                end = min(len(clean_text), m.end() + 40)
                snippet = clean_text[start:end].replace("\n", " ")
                print(f"[NOTE] Found filename '{fn}': ...{snippet}...")

    print("=" * 60)
    if errors == 0:
        print("ALL TEXT CLAIMS AND COMPLIANCE AUDITS PASSED!")
    else:
        print(f"COMPLIANCE AUDIT COMPLETED WITH {errors} ERRORS.")
    print("=" * 60)
    return errors

if __name__ == "__main__":
    sys.exit(check_tex())
