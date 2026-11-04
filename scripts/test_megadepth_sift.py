import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
import megadepth as md

try:
    sift = cv2.SIFT_create(nfeatures=2000)
except AttributeError:
    from cv_compat import make
    sift = make("SIFT", nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_L2)

pairs = md.list_pairs()[::15][:100]          # 100 pairs spread over all 5 files
errs, t0 = [], time.time()
for n, (f, i, j) in enumerate(pairs):
    img0, img1, K0, K1, T = md.load_pair(f, i, j)
    k0, d0 = sift.detectAndCompute(img0, None)
    k1, d1 = sift.detectAndCompute(img1, None)
    err = np.inf
    if d0 is not None and d1 is not None and len(k0) > 8 and len(k1) > 8:
        good = [m for m, nn in bf.knnMatch(d0, d1, k=2) if m.distance < 0.75 * nn.distance]
        if len(good) >= 8:
            p0 = np.float64([k0[m.queryIdx].pt for m in good])
            p1 = np.float64([k1[m.trainIdx].pt for m in good])
            est = md.estimate_pose(p0, p1, K0, K1)
            if est is not None:
                err = md.pose_error(T, *est)
    errs.append(err)
    if n % 20 == 0:
        print(f"pair {n}: err={err:.2f} deg ({time.time() - t0:.0f}s)")

errs = np.array(errs)
print("\npairs:", len(errs), "| failed (inf):", int(np.isinf(errs).sum()),
      "| median error (deg):", round(float(np.median(errs)), 2))
a5, a10, a20 = md.pose_auc(errs)
print(f"SIFT-2000 AUC@5/10/20 deg: {a5*100:.1f} / {a10*100:.1f} / {a20*100:.1f}")