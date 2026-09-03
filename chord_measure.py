#!/usr/bin/env python3
"""
chords.py -- length and width sampled ACROSS the coupon, not just the single
             fitted figure gauge.py reports.

    python3 chords.py --image shot.png --auto-roi --thickness 5.5
    python3 chords.py --image shot.png --auto-roi --thickness 5.5 -n 30 --csv prof.csv
    python3 chords.py --image shot.png --auto-roi --thickness 5.5 --band 0

Imports gauge.py for capture, ROI, edge finding and calibration. Nothing here
changes gauge.py -- this is a second view of the same scan data.

WHAT A CHORD IS
    Every row of the scan already yields a left and a right subpixel edge. Their
    separation is the coupon's extent along that row -- one horizontal chord.
    gauge.py fits two lines through all of them and reports one number, which
    throws the variation away. This keeps it.

        rows    -> horizontal chords -> the long dimension at each height
        columns -> vertical chords   -> the short dimension at each position

SINGLE ROW OR A BAND
    One row carries the full per-row noise, ~0.04 px, about 2.5 um at 58 um/px.
    Usable, but 20x worse than the fitted line, because a line through N rows
    averages the noise down as 1/sqrt(N).

    A band recovers most of that: averaging over B rows divides the noise by
    sqrt(B) while staying localised to B rows of the coupon. The band is
    specified in MILLIMETRES so it means the same thing at any resolution or
    standoff. Default 3 mm -- far finer than pressed wood varies.

    --band 0 disables banding and gives the raw per-row chords.

WHY THE ENDS GET TRIMMED
    On a tilted coupon, a horizontal line near the top does not cross both long
    edges -- it clips a corner, and the two "edges" it finds are a short corner
    chord. Those chords are real measurements of the wrong thing. The first and
    last few percent of each axis are therefore dropped. Raise --trim if the
    profile still turns down at both ends; that shape is the tell.

TILT STILL COMES FROM THE FITTED LINES
    A horizontal chord is only the true length if the coupon is square to the
    frame; tilted by theta it is longer by 1/cos(theta). A single chord has no
    slope to correct with, so the angle comes from the global edge fit. That is
    what the fit is good for, which is why this augments gauge.py rather than
    replacing it.
"""

import argparse
import csv
from typing import NamedTuple

import cv2
import numpy as np

import gauge as G


class Chord(NamedTuple):
    pos_px: float       # position along the scan axis, ROI pixels
    pos_mm: float       # same, mm from the first chord
    span_px: float      # tilt-corrected extent between the two edges
    span_mm: float
    n_rows: int         # rows averaged into this chord
    scatter_px: float   # spread of the raw per-row spans inside the band


class Profile(NamedTuple):
    chords: list
    tilt_deg: float
    mmpx: float
    label: str
    shape: dict = None      # per-edge waviness / roughness split

    def spans(self):
        return np.array([c.span_mm for c in self.chords])

    def summary(self):
        s = self.spans()
        return {"n": len(s), "min": float(s.min()), "max": float(s.max()),
                "mean": float(s.mean()),
                "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "range": float(s.max() - s.min())}


def _robust_mean(v):
    """Mean after dropping points more than 3 MAD from the median.

    A band is a small sample, so one bad row -- a nick, a dust speck -- shifts a
    plain mean noticeably. Rejecting on median absolute deviation rather than
    standard deviation matters because with only a few dozen points a single
    outlier inflates sigma enough to survive its own test.
    """
    v = np.asarray(v, float)
    if v.size < 4:
        return float(v.mean()), v.size
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    keep = np.abs(v - med) < 3.0 * max(mad, 1e-4)
    if keep.sum() < 3:
        return float(med), v.size
    return float(v[keep].mean()), int(keep.sum())


def decompose(pos, edge, order=3):
    """Split an edge trace into straight, wavy and rough components.

    These are three physically different things and lumping them into one RMS
    hides which one you have:

      straight  -- the best-fit line: the coupon's nominal edge
      waviness  -- a smooth low-order departure from it: bow, saw wander, a
                   press that closed unevenly. Real geometry, and it is what
                   makes a single width number meaningless.
      roughness -- what is left: fibre tear-out. Random, so the fit averages it
                   away and it costs almost nothing.

    A 4 mm width variation that is all waviness means the specimen is genuinely
    that shape. The same number arising as roughness would mean a ragged edge
    around a true rectangle. Different problems, different fixes.
    """
    resid = edge - np.polyval(np.polyfit(pos, edge, 1), pos)
    wavy = np.polyval(np.polyfit(pos, resid, order), pos)
    rough = resid - wavy
    return {"wave_rms": float(np.sqrt(np.mean(wavy ** 2))),
            "wave_ptp": float(wavy.max() - wavy.min()),
            "rough_rms": float(np.sqrt(np.mean(rough ** 2)))}


def profile_axis(pos, lo, hi, cfg, mmpx, n_chords, band_mm, trim_frac, label,
                 order=3):
    """Build one set of chords from a scan's (position, low edge, high edge).

    The perpendicular correction is taken from the LOCAL direction of the
    specimen's centreline, not from one global slope. On a curved or wavy edge
    the two differ, and using a global slope quietly reports the wrong quantity:
    the chord is only the true width when it is corrected by the cosine of the
    angle at THAT point.
    """
    centre = 0.5 * (lo + hi)
    # A low-order polynomial through the centreline: smooth enough to be a stable
    # local direction, flexible enough to follow real bow. Differentiated
    # analytically rather than by finite differences, which would amplify the
    # per-row noise into a useless slope.
    cpoly = np.polyfit(pos, centre, order)
    dpoly = np.polyder(cpoly)
    mloc = np.polyval(dpoly, pos)
    raw = (hi - lo) / np.sqrt(1.0 + mloc ** 2)

    shape = {"lo": decompose(pos, lo, order), "hi": decompose(pos, hi, order)}

    span = pos.max() - pos.min()
    a = pos.min() + trim_frac * span
    b = pos.max() - trim_frac * span
    if b <= a:
        raise RuntimeError("nothing left after trimming; lower --trim")

    half = 0.5 * max(band_mm / mmpx, 0.0) if mmpx > 0 else 0.0
    centres = np.linspace(a, b, n_chords) if n_chords > 1 else np.array([0.5 * (a + b)])

    out = []
    for c in centres:
        sel = (np.abs(pos - c) <= half) if half > 0 else (np.abs(pos - c) < 0.5)
        if sel.sum() == 0:                       # band narrower than the row pitch
            sel = np.zeros(pos.size, bool)
            sel[int(np.argmin(np.abs(pos - c)))] = True
        vals = raw[sel]
        mean_px, used = _robust_mean(vals)
        out.append(Chord(
            pos_px=float(c), pos_mm=float((c - a) * mmpx),
            span_px=mean_px, span_mm=mean_px * mmpx, n_rows=used,
            scatter_px=float(vals.std(ddof=1)) if vals.size > 1 else 0.0))

    m_mid = float(np.polyval(dpoly, 0.5 * (a + b)))
    return Profile(out, float(np.degrees(np.arctan(m_mid))), mmpx, label, shape)


def measure_chords(img, cfg, thickness, n_chords, band_mm, trim_frac, order=3):
    """Both directions. Returns (long_axis_profile, short_axis_profile)."""
    x0, y0, x1, y1 = cfg.roi
    H, W = img.shape
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise ValueError(f"roi {cfg.roi} does not fit inside a {W}x{H} frame")
    roi = img[y0:y1, x0:x1]
    mmpx = cfg.mmPerPx(thickness) if cfg.focalPx > 0 else 0.0

    ry, rl, rr = G.scan(roi, cfg)                                  # rows
    cy, cl, cr = G.scan(np.ascontiguousarray(roi.T), cfg)          # columns

    # Drop scan lines that clip a corner instead of crossing both end faces --
    # the same trap gauge.measure() guards against. It bites harder here: a bowed
    # edge extends the specimen's bounding box beyond the flat part, so the extra
    # lines all sit in the corner region and their chords are pure artefact.
    if cfg.trimCorners:
        for name, (pos, lo, hi) in (("rows", (ry, rl, rr)), ("cols", (cy, cl, cr))):
            a_, b_ = G.valid_band(pos, lo, hi)
            if b_ - a_ < 20:
                raise RuntimeError(f"only {b_-a_} {name} cross both end faces - "
                                   f"too rotated, or the edges are too curved")
            if name == "rows":
                ry, rl, rr = pos[a_:b_+1], lo[a_:b_+1], hi[a_:b_+1]
            else:
                cy, cl, cr = pos[a_:b_+1], lo[a_:b_+1], hi[a_:b_+1]

    p_row = profile_axis(ry, rl, rr, cfg, mmpx, n_chords, band_mm, trim_frac,
                         "row", order)
    p_col = profile_axis(cy, cl, cr, cfg, mmpx, n_chords, band_mm, trim_frac,
                         "col", order)

    # Name by magnitude, as gauge.py does, so the answer does not depend on which
    # way round the coupon was laid down.
    if p_row.spans().mean() >= p_col.spans().mean():
        return p_row._replace(label="length"), p_col._replace(label="width")
    return p_col._replace(label="length"), p_row._replace(label="width")


def sparkline(p):
    """Terminal profile. Over SSH this is faster to read than opening a plot."""
    s = p.spans()
    rng = float(s.max() - s.min())
    if rng < 1e-9:
        return "  (flat)"
    blocks = " .:-=+*#%@"
    idx = ((s - s.min()) / rng * (len(blocks) - 1)).round().astype(int)
    return "  " + "".join(blocks[i] for i in idx)


def report(p, unit_mm):
    """unit_mm also selects the units of the shape breakdown."""
    st = p.summary()
    u = "mm" if unit_mm else "px"
    k = 1.0 if unit_mm else 1.0
    print(f"\n  {p.label.upper()}  ({st['n']} chords, tilt {p.tilt_deg:+.3f} deg)")
    if unit_mm:
        print(f"    min {st['min']:9.4f}   max {st['max']:9.4f}   "
              f"mean {st['mean']:9.4f} {u}")
        print(f"    range {st['range']*1000:8.1f} um   sd {st['sd']*1000:7.1f} um")
    else:
        sp = np.array([c.span_px for c in p.chords])
        print(f"    min {sp.min():9.3f}   max {sp.max():9.3f}   "
              f"mean {sp.mean():9.3f} px   (uncalibrated)")
    print(f"    rows per chord {p.chords[0].n_rows}   "
          f"typical within-band scatter {np.median([c.scatter_px for c in p.chords]):.4f} px")
    if p.shape:
        k = p.mmpx * 1000 if unit_mm else 1.0
        uu = "um" if unit_mm else "px"
        for side, sh in (("edge A", p.shape["lo"]), ("edge B", p.shape["hi"])):
            print(f"    {side}: waviness {sh['wave_ptp']*k:7.1f} {uu} p-p"
                  f"   roughness {sh['rough_rms']*k:6.1f} {uu} rms")
    print(sparkline(p))
    # A profile that dips at both ends and peaks in the middle usually means the
    # trim is too small and the end chords are clipping corners, not that the
    # coupon is barrel-shaped.
    # A chord that clips a corner is a real measurement of the wrong thing, and it
    # only takes ONE to drag the mean. Compare each end against the robust middle
    # rather than requiring both ends to be low.
    s = p.spans() if unit_mm else np.array([c.span_px for c in p.chords])
    if len(s) >= 5:
        mid = float(np.median(s[len(s)//4: 3*len(s)//4]))
        for lbl, v in (("first", s[0]), ("last", s[-1])):
            if mid - v > 0.002 * mid:
                print(f"    note: {lbl} chord low by {(mid-v)/mid*100:.2f}% "
                      f"-- corner clip, raise --trim")
        good = s[np.abs(s - mid) < 0.05 * mid]
        if len(good) < len(s):
            print(f"    excluding {len(s)-len(good)} corner-clipped chord(s): "
                  f"mean {good.mean():.4f}, range {(good.max()-good.min())*1000:.1f} um")


def write_csv(path, profs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "index", "pos_px", "pos_mm", "span_px", "span_mm",
                    "n_rows", "scatter_px", "tilt_deg"])
        for p in profs:
            for i, c in enumerate(p.chords):
                w.writerow([p.label, i, f"{c.pos_px:.2f}", f"{c.pos_mm:.4f}",
                            f"{c.span_px:.4f}", f"{c.span_mm:.4f}",
                            c.n_rows, f"{c.scatter_px:.4f}", f"{p.tilt_deg:.4f}"])
    print(f"\n  wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1")
    p.add_argument("--auto-roi", action="store_true")
    p.add_argument("--thickness", type=float, required=True, help="specT, mm")
    p.add_argument("-n", "--chords", type=int, default=20, help="chords per axis")
    p.add_argument("--band", type=float, default=3.0,
                   help="band height in mm averaged into each chord; 0 = single row")
    p.add_argument("--order", type=int, default=3,
                   help="polynomial order for the centreline and the waviness "
                        "split. 1 = assume straight edges; 3 handles bow; higher "
                        "follows more shape but starts fitting noise.")
    p.add_argument("--trim", type=float, default=0.05,
                   help="fraction dropped from each end, where chords clip corners")
    p.add_argument("--csv", metavar="FILE")
    p.add_argument("--eff-dist", type=float)
    p.add_argument("--focal-px", type=float)
    p.add_argument("--calib-file", default="calibration.json")
    a = p.parse_args()

    cfg = G.Config()
    cfg.specT = a.thickness
    if a.eff_dist:
        cfg.effDist = a.eff_dist
    if a.focal_px:
        cfg.focalPx = a.focal_px
    else:
        G.load_calib(a.calib_file, cfg)

    img = G.load(a.image)
    if a.roi:
        cfg.roi = tuple(int(v) for v in a.roi.split(","))
    elif a.auto_roi:
        cfg.roi = G.auto_roi(img)
        print(f"auto ROI {cfg.roi}  (image {img.shape[1]}x{img.shape[0]})")

    if cfg.focalPx > 0:
        cfg.focalPx = G.scale_focal(cfg.focalPx, 0, img.shape[1])

    band = a.band if cfg.focalPx > 0 else 0.0
    if a.band > 0 and cfg.focalPx <= 0:
        print("  uncalibrated, so --band has no mm to work from: using single rows")

    long_p, short_p = measure_chords(img, cfg, a.thickness, a.chords, band,
                                     a.trim, a.order)
    unit_mm = cfg.focalPx > 0
    for pr in (long_p, short_p):
        report(pr, unit_mm)
    if unit_mm:
        print(f"\n  scale {cfg.mmPerPx(a.thickness)*1000:.4f} um/px")
    if a.csv:
        write_csv(a.csv, (long_p, short_p))


if __name__ == "__main__":
    main()