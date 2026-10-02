import cv2, types

targets = ("AKAZE", "BRISK", "KAZE")

def scan(mod, path, seen, depth=0):
    if id(mod) in seen or depth > 2:
        return
    seen.add(id(mod))
    for n in dir(mod):
        if any(t in n.upper() for t in targets):
            print(f"{path}.{n}")
        try:
            sub = getattr(mod, n)
        except Exception:
            continue
        if isinstance(sub, types.ModuleType) and sub.__name__.startswith("cv2"):
            scan(sub, f"{path}.{n}", seen, depth + 1)

scan(cv2, "cv2", set())
print("done")