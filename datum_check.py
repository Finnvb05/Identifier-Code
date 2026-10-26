import sys, time
import numpy as np, cv2
import gauge as G
from pi_dual import Cam, Role

VIEWS = {"bottom": (1, "ref_bottom.npy", "calib_bottom.json", 775.0),
         "side":   (0, "ref_side.npy",   "calib_side.json",   300.0)}

name = sys.argv[1] if len(sys.argv) > 1 else "side"
index, ref_file, calib_file, eff = VIEWS[name]

role = Role(name, eff, None, 3.0, calib_file, 0.01, ref_file, "below")
role.cam = Cam(index, exposure=None)
try:
    time.sleep(1.5)
    img = role.grab()
finally:
    role.cam.close()

print(f"\n=== {name} (sensor {index}) ===  frame {img.shape[1]}x{img.shape[0]}")
box = role.find_roi(img)
cfg = G.replace(role.cfg, roi=box)
x0, y0, x1, y1 = box
print(f"ROI x {x0}..{x1}  y {y0}..{y1}   ({x1-x0} x {y1-y0})")

roi, rref = G._roi(img, cfg), G._roi(role.ref.ref, cfg)
T = np.ascontiguousarray
small_arr, small_ref = T(roi.T), T(rref.T)
print(f"scan line length {small_arr.shape[1]} px (ROI height)")

gw = G.grad_width(cv2.GaussianBlur(small_arr, (0,0), cfg.sigma), cfg)
hw = int(np.clip(round(cfg.halfwinK*gw), 3, 30)) if gw else 6
print(f"edge width {gw if gw else float('nan'):.2f} px RMS -> window +/-{hw} px")

dp, dv = G._edge_points(small_ref, cfg, False, hw)
print(f"\nREFERENCE datum: {dp.size} of {small_arr.shape[0]} columns (need >=20)")
if dp.size: print(f"  at ROI y {dv.min():.0f}..{dv.max():.0f} = frame row "
                  f"{y0+dv.min():.0f}..{y0+dv.max():.0f}")
if dp.size < 20: print("  -> FAILS HERE: ROI misses the fixture edge"); sys.exit()
dline = G.fit(dp, dv)

cp, cv = G._edge_points(small_arr, cfg, False, hw)
print(f"\nLIVE free edge:  {cp.size} of {small_arr.shape[0]} columns")
if cp.size == 0:
    print(f"  -> FAILS HERE: no edge on any column. Coupon top is probably")
    print(f"     less than {hw} px below the ROI top, so the window overruns.")
    sys.exit()
print(f"  at ROI y {cv.min():.0f}..{cv.max():.0f} = frame row "
      f"{y0+cv.min():.0f}..{y0+cv.max():.0f}")
print(f"  clearance above ROI top: {cv.min():.0f} px (need >= {hw})")

sep = dline.slope*cp + dline.icept - cv
print(f"\nCLEARANCE (datum_y - edge_y > 5):")
print(f"  separation {sep.min():+.1f} .. {sep.max():+.1f} px, median {np.median(sep):+.1f}")
print(f"  passing: {int((sep>5).sum())} (need >=20)")
