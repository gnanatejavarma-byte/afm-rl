import glob, cv2, torch, inspect
import kornia.feature as KF

path = sorted(glob.glob("data/hpatches/*/1.ppm"))[0]
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
t = torch.from_numpy(img)[None, None].float() / 255.0
print("image:", path, img.shape)

print("XFeat public methods:", [a for a in dir(KF.XFeat) if not a.startswith("_")])
print("from_pretrained signature:", inspect.signature(KF.XFeat.from_pretrained))

def run(model, label):
    for name in ("detectAndCompute", "extract", "detect_and_compute"):
        if hasattr(model, name):
            with torch.no_grad():
                out = getattr(model, name)(t)
            o = out[0] if isinstance(out, (list, tuple)) else out
            kp = o["keypoints"] if isinstance(o, dict) else o.keypoints
            print(f"{label}: via {name}() -> {len(kp)} keypoints")
            return
    print(f"{label}: no known extraction method found")

run(KF.XFeat().eval(), "XFeat() plain")
run(KF.XFeat.from_pretrained().eval(), "XFeat.from_pretrained()")

m2 = KF.ALIKED.from_pretrained("aliked-n16").eval()
with torch.no_grad():
    out2 = m2(t)
f = out2[0]
print("ALIKED attributes:", [a for a in dir(f) if not a.startswith("_")])