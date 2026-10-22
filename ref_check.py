#!/usr/bin/env python3
"""Show what the measurement actually sees after flat-fielding.

The live preview streams the RAW lores frame, but measurement runs on the main
frame with the reference divided out. When those two disagree you are debugging
blind. This grabs one frame, applies the reference, and writes out every stage
plus the numbers auto_roi is deciding on.

    python3 ref_check.py bottom
    python3 ref_check.py side

Writes chk_<view>_{raw,ref,mask,flat}.png, downscaled for sending.
"""
import sys
import numpy as np
import cv2
import gauge as G
from pi_dual import Cam

VIEWS = {"bottom": (1, "ref_bottom.npy"), "side": (0, "ref_side.npy")}


def save(name, arr, width=1200):
    """Normalise to 0-255 and downscale, so the file is small enough to send."""
    a = np.asarray(arr, np.float32)
    lo, hi = float(a.min()), float(a.max())
    span = hi - lo if hi > lo else 1.0
    img = ((a - lo) / span * 255).astype(np.uint8)
    if img.shape[1] > width:
        k = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * k)), cv2.INTER_AREA)
    cv2.imwrite(name, img)
    print(f"  wrote {name}   (displayed range {lo:.0f}..{hi:.0f} DN)")


def projections(img):
    """Reproduce auto_roi's decision, and say what it is looking at."""
    H, W = img.shape
    sm = cv2.GaussianBlur(img, (0, 0), 2.0)
    for axis, g, n, label in ((0, cv2.Sobel(sm, cv2.CV_32F, 1, 0, 3), W, "X (left/right)"),
                              (1, cv2.Sobel(sm, cv2.CV_32F, 0, 1, 3), H, "Y (top/bottom)")):
        fall = np.clip(-g, 0, None).sum(axis=axis)
        rise = np.clip(g, 0, None).sum(axis=axis)
        edge = max(3, n // 50)
        fall[:edge] = fall[n - edge:] = 0
        rise[:edge] = rise[n - edge:] = 0
        base = max(float(np.median(fall[edge:n - edge])),
                   float(np.median(rise[edge:n - edge])), 1e-6)
        fi, ri = int(np.argmax(fall)), int(np.argmax(rise))
        print(f"  {label:<16} n={n}  border zone +/-{edge} px")
        print(f"      strongest FALL at {fi:5d}  ({fall[fi]/base:6.1f} x median)")
        print(f"      strongest RISE at {ri:5d}  ({rise[ri]/base:6.1f} x median)")
        if ri > fi:
            print(f"      -> that pair spans {ri - fi} px")
        else:
            print(f"      -> RISE is LEFT of FALL: not a dark object")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in VIEWS:
        raise SystemExit(f"usage: ref_check.py {'|'.join(VIEWS)}")
    view = sys.argv[1]
    index, ref_file = VIEWS[view]

    cam = Cam(index, exposure=None)
    try:
        import time
        time.sleep(1.5)                      # let the sensor settle
        raw = cam.grab()
    finally:
        cam.close()

    print(f"\n=== {view} (sensor {index}) ===")
    print(f"  frame {raw.shape[1]} x {raw.shape[0]}   "
          f"min {raw.min():.0f}  max {raw.max():.0f}  "
          f"5th {np.percentile(raw,5):.0f}  95th {np.percentile(raw,95):.0f}")
    if raw.max() >= 255:
        pct = float((raw >= 254).mean() * 100)
        print(f"  WARNING {pct:.2f}% of the frame is saturated")
    save(f"chk_{view}_raw.png", raw)

    try:
        refarr = np.load(ref_file)
    except Exception as e:
        print(f"  no reference ({e}) - stopping after the raw frame")
        print("\n--- auto_roi on the RAW frame ---")
        projections(raw)
        return

    ref = G.Reference(refarr)
    print(f"\n  reference {ref_file}")
    print(f"    background      {ref.bright:.0f} DN")
    print(f"    opaque          {ref.masked_frac*100:.1f}% of frame")
    ys, xs = np.where(ref.mask)
    if ys.size:
        print(f"    mask bbox       x {xs.min()}..{xs.max()}  y {ys.min()}..{ys.max()}")
        print(f"    mask rows       {ys.min()}..{ys.max()} of {ref.mask.shape[0]}")
    save(f"chk_{view}_ref.png", refarr)
    save(f"chk_{view}_mask.png", ref.mask.astype(np.uint8) * 255)

    flat, touch = ref.apply(raw)
    print(f"\n  after flat-fielding")
    print(f"    touch           {touch} px of the 3 px ring are dark (limit 20)")
    print(f"    min {flat.min():.0f}  max {flat.max():.0f}  "
          f"5th {np.percentile(flat,5):.0f}  95th {np.percentile(flat,95):.0f}")
    save(f"chk_{view}_flat.png", flat)

    print("\n--- auto_roi on the FLAT-FIELDED frame (what it really decides on) ---")
    projections(flat)
    try:
        box = G.auto_roi(flat)
        print(f"\n  auto_roi -> {box}")
    except Exception as e:
        print(f"\n  auto_roi FAILED: {e}")

    print("\n--- auto_roi on the RAW frame, for comparison ---")
    projections(raw)
    try:
        print(f"\n  auto_roi -> {G.auto_roi(raw)}")
    except Exception as e:
        print(f"\n  auto_roi FAILED: {e}")


if __name__ == "__main__":
    main()