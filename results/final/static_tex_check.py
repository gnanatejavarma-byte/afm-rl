"""
Static analysis and linting script for paper/afm_rl_paper.tex.
Performs comprehensive structural verification without running pdflatex.
"""

import re
import sys
from pathlib import Path

def check_latex_syntax(tex_path):
    print(f"Running static analysis on {tex_path}...")
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.splitlines()

    errors = 0
    warnings = 0

    # 1. Environment balance
    env_stack = []
    env_pattern = re.compile(r"\\(begin|end)\{([a-zA-Z*]+)\}")
    for idx, line in enumerate(lines, 1):
        # Ignore comments
        clean_line = re.sub(r"(?<!\\)%.*", "", line)
        for m in env_pattern.finditer(clean_line):
            cmd, env_name = m.groups()
            if cmd == "begin":
                env_stack.append((env_name, idx))
            elif cmd == "end":
                if not env_stack:
                    print(f"  [ERROR] Line {idx}: Unmatched \\end{{{env_name}}}")
                    errors += 1
                else:
                    last_env, start_idx = env_stack.pop()
                    if last_env != env_name:
                        print(f"  [ERROR] Line {idx}: Mismatched \\end{{{env_name}}}, expected \\end{{{last_env}}} opened at line {start_idx}")
                        errors += 1

    if env_stack:
        for env_name, start_idx in env_stack:
            print(f"  [ERROR] Line {start_idx}: Unclosed \\begin{{{env_name}}}")
            errors += 1

    # 2. Check labels and refs
    labels = set(re.findall(r"\\label\{([^}]+)\}", content))
    refs = re.findall(r"\\ref\{([^}]+)\}", content)
    for r in refs:
        if r not in labels:
            print(f"  [ERROR] Undefined \\ref{{{r}}}")
            errors += 1

    # 3. Check bibitems and cites
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", content))
    cites_raw = re.findall(r"\\cite\{([^}]+)\}", content)
    cites = set()
    for c in cites_raw:
        for item in c.split(","):
            cites.add(item.strip())

    for c in cites:
        if c not in bibitems:
            print(f"  [ERROR] Undefined citation \\cite{{{c}}}")
            errors += 1

    for b in bibitems:
        if b not in cites:
            print(f"  [WARNING] Unused bibitem: '{b}'")
            warnings += 1

    # 4. Check inline math delimiter balance per line
    for idx, line in enumerate(lines, 1):
        clean_line = re.sub(r"(?<!\\)%.*", "", line)
        # Count non-escaped $
        dollar_count = len(re.findall(r"(?<!\\)\$", clean_line))
        if dollar_count % 2 != 0:
            print(f"  [WARNING] Line {idx}: Odd number of inline math delimiters ($): '{clean_line[:60]}...'")
            warnings += 1

    # 5. Check tabular column counts
    tabular_blocks = re.findall(r"\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}", content, re.DOTALL)
    for col_spec, tab_body in tabular_blocks:
        col_count = len(re.findall(r"[lcrpm]", col_spec))
        for r_idx, row in enumerate(tab_body.strip().split(r"\\")):
            row_clean = re.sub(r"%.*", "", row).strip()
            if not row_clean or r"\hline" in row_clean and not any(c in row_clean for c in ["&", "\\textbf"]):
                continue
            if r"\multicolumn" in row_clean:
                continue
            amps = row_clean.count("&")
            if amps > 0 and amps != col_count - 1:
                print(f"  [WARNING] Tabular row has {amps + 1} columns, expected {col_count}: '{row_clean[:50]}...'")
                warnings += 1

    print(f"Static check complete. Errors: {errors}, Warnings: {warnings}")
    return errors

if __name__ == "__main__":
    errs = check_latex_syntax("paper/afm_rl_paper.tex")
    if errs > 0:
        sys.exit(1)
    else:
        print("ALL STATIC CHECKS PASSED PERFECTLY!")
        sys.exit(0)
