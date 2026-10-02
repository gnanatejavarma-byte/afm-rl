import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hpatches import load_split, load_pair
import pipeline_utils as pu

val = load_split()["val"]
seq_i = next(s for s in val if s.startswith("i_"))
seq_v = next(s for s in val if s.startswith("v_"))
tests = [(seq_i, 3), (seq_v, 5)]
GRID = [100, 500, 1000, 2000, 4000]

for seq, k in tests:
    img1, img2, H = load_pair(seq, k)
    key1, key2 = (seq, 1), (seq, k)
    print(f"\n=== {seq} pair 1-{k}  image size {img1.shape[1]}x{img1.shape[0]} ===")

    print("Pool build (one-time cost per image+preset):")
    for p in pu.PRESET_NAMES:
        a, ta = pu.get_pool(img1, p, key1)
        b, tb = pu.get_pool(img2, p, key2)
        print(f"  {p:11s} pool1={len(a['pts']):5d} pool2={len(b['pts']):5d}  build={1000*(ta+tb):8.1f} ms")

    print(f"\n{'preset':11s} {'match':5s} {'N_req':>5s} {'kp1':>5s} {'kp2':>5s} {'matches':>7s} {'match_ms':>9s}")
    for p in pu.PRESET_NAMES:
        for m in pu.VALID_MATCHERS[pu.PRESETS[p][2]]:
            for n in GRID:
                try:
                    r = pu.run_pipeline(img1, img2, p, n, matcher=m, key1=key1, key2=key2)
                    flag = "" if r["pool_s"] == 0.0 else "  <-- cache MISS"
                    print(f"{p:11s} {m:5s} {n:5d} {r['n_kp1']:5d} {r['n_kp2']:5d} "
                          f"{r['n_matches']:7d} {r['match_s']*1000:9.1f}{flag}")
                except Exception as e:
                    print(f"{p:11s} {m:5s} {n:5d}  FAIL: {type(e).__name__}: {e}")