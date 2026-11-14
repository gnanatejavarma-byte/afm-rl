
import glob

files = glob.glob('results/**/*.py', recursive=True) + glob.glob('results/**/*.md', recursive=True)
for f in files:
    with open(f, encoding='utf-8', errors='ignore') as fp_in:
        txt = fp_in.read()
    if 'ALIKED-1000' in txt and 'SIFT' in txt:
        for l in txt.splitlines():
            if 'C4' in l or 'C2' in l or 'C6' in l:
                print(f, ':', l.strip()[:100])
