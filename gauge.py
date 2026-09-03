#!/usr/bin/env python3
r"""
Backlit silhouette gauge -- coupon length and width from one image.
Variable names follow PaperSteel_Onshape_Variables.csv so the code and the CAD
parametric model cannot drift apart.

PHYSICAL SETUP
--------------
        camera            armZ    = zSpec + workDist + camStack
          |
          |  workDist             lens front face -> specimen plane
          v
    [==== coupon ====]    specT   = specTden (5.5) or specTraw (27)
    ------------------    zSpec   = zPanBot + panT + ribH = 62 mm  <- MEASUREMENT DATUM
        backlight         zBacklight = 14 mm

THE MEASUREMENT CHAIN
---------------------
    objDist   = lensF * (1 + 1/mag)      principal plane -> specimen PLANE
    sObj      = objDist - specT          -> silhouette-forming TOP face
    mmPerPx   = (sObj - lensF) / focalPx
    size_mm   = size_px * mmPerPx

Why (sObj - lensF) and not sObj:
    A pinhole scales as s. A real lens scales as (s - f), because the transverse
    magnification is m = f/(s - f). With lensF = 12 mm and objDist = 462 mm the two
    differ by 2.66% in absolute scale -- which calibration absorbs harmlessly, since
    it only rescales focalPx.

    What calibration does NOT absorb is the thickness correction, because that is a
    DIFFERENCE of scales at two standoffs. Measured on this geometry:

        specT =  5.5 mm : correction is 12210 ppm, pinhole errs by  +321 ppm
        specT = 27   mm : correction is 59940 ppm, pinhole errs by +1655 ppm

    Against a length budget of 125 ppm (25 um on specL = 200 mm), the pinhole form
    alone is 2.6x over at the densified thickness and 13x over at the raw thickness.
    Hence lensF appears explicitly in the scale law.

Why specT subtracts at all, and subtracts in full:
    The dark region is the coupon outline seen from the lens. A block with vertical
    walls has two candidate outlines, its top face and its bottom face; the top face
    is nearer the lens, subtends a larger angle and projects larger, so the bottom
    face outline falls strictly inside it and is never visible. The outline measured
    is therefore the TOP face, one full thickness above zSpec.

    Not a small correction here: 5.5 mm at objDist 462 mm is 1.2%, which on
    couponW = 21.25 mm is 259 um. On a raw 27 mm panel it is 1.27 mm.

WHY THE ABSOLUTE STANDOFF BARELY MATTERS
----------------------------------------
    Calibration solves  focalPx = W_cal_px * (s_cal - lensF) / W_cal_mm. Substituting
    back collapses the chain to:

        W_mm = W_cal_mm * (W_px / W_cal_px) * (sObj - lensF) / (s_cal - lensF)

    Only the RATIO survives, so the entrance pupil position never has to be known and
    objDist need only be REPEATABLE between calibration and measurement, not known
    absolutely. Refocusing changes both f and the pupil position and voids this
    silently -- focus once, lock the ring, recalibrate if disturbed.

    Sanity check on any calibration: focalPx should land near lensF / pxPitch
    = 12 / 0.00155 = 7742 px. A large discrepancy means the built geometry is not
    what the CAD says it is.

WHY NOT THRESHOLD TO BLACK AND WHITE
------------------------------------
    A threshold snaps each edge to a whole pixel, costing the full mmPerPx (58.2 um
    here) on both edges. The greyscale gradient-centroid path below reaches ~0.02 px,
    i.e. ~1.2 um. Binarise for a visual check if you like; never measure off it.

    pip install numpy opencv-python requests
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from typing import NamedTuple, Optional

import cv2
import numpy as np
import requests


# ============================================================== config ======

MAX_RESID_FRAC = 0.01           # default ceiling on the edge fit residual,
                                # as a fraction of the dimension being measured


@dataclass
class Config:
    """Names match PaperSteel_Onshape_Variables.csv. Inputs carry the CSV values;
    DERIVED quantities are properties computed by the CSV formulas, so changing an
    input propagates instead of leaving a stale duplicate."""

    host: str = "192.168.1.50"

    # --- specimen envelope (CSV) ---------------------------------------------
    specL: float = 200.0            # longest specimen handled on machine 1
    specW: float = 100.0            # widest specimen
    specTraw: float = 27.0          # thickness as sawn, before the bath
    specTden: float = 5.5           # thickness after pressing
    couponN: int = 4                # tensile coupons cut per panel
    couponGap: float = 5.0          # saw kerf + trim allowance between coupons

    # This coupon's MEASURED thickness -- not specTden, which is only the nominal.
    # At objDist 462 mm, 1 mm of error here is 2200 ppm: 17x the length budget.
    specT: float = 5.5

    # --- datum stack (CSV) ---------------------------------------------------
    zPanBot: float = 44.0           # underside of the pan frame above the base plate
    panT: float = 12.0              # pan frame plate thickness
    ribH: float = 6.0               # rib height above the pan plate
    zBacklight: float = 14.0        # top face of the backlight panel

    # --- optics (CSV) --------------------------------------------------------
    fovMargin: float = 1.18         # field of view as a multiple of specimen length
    sensorW: float = 6.287          # IMX477 active area, long axis
    sensorH: float = 4.712          # IMX477 active area, short axis
    pixelsL: int = 4056             # pixels on the long axis
    lensF: float = 12.0             # lens focal length
    lensPP: float = 15.5            # lens front face to its principal plane
    camStack: float = 45.0          # lens front face up to the camera arm underside

    # --- calibration ---------------------------------------------------------
    focalPx: float = 0.0            # from --calibrate. 0 = uncalibrated and refuses
                                    # to measure. Expect ~focalPxNominal.
    calibSObj: float = 0.0          # sObj at calibration, recorded to detect drift

    # Effective distance A = objDist - lensF, i.e. platform-to-equivalent-pinhole with
    # the focal length already folded in. 0 = derive it from the CAD optics above.
    # --calib-stack MEASURES it, which is what you want on any rig where lensF and
    # lensPP are unknown (an ESP32-CAM module, or any lens without a datasheet).
    effDist: float = 0.0

    # --- region of interest --------------------------------------------------
    # Must contain the coupon and nothing else: the row scan assumes one dark object
    # per line. --auto-roi will locate it.
    roi: tuple[int, int, int, int] = (400, 300, 1200, 900)

    # --- sensor --------------------------------------------------------------
    exposure: int = 300             # manual AEC, 0..1200. Aim the bright field near
                                    # 200 DN. Measured cost of clipping: -334
                                    # millipixels at 270 DN nominal, -879 at 300. It
                                    # truncates the gradient on the bright side only,
                                    # so both edges walk INWARD and the coupon reads
                                    # UNDERSIZE -- silently, with good repeatability.
    framesize: int = 13             # 13 = UXGA. 17 = QXGA, the full 4056 px the CSV
                                    # assumes; needs PSRAM.
    jpegQuality: int = 4            # LOWER = better. JPEG ringing lands on your edges.

    # --- edge detection ------------------------------------------------------
    sigma: float = 1.0              # Gaussian pre-smooth, px. Symmetric -> no bias.
    halfwin: int = 0                # centroid window, px either side. 0 = size it from
                                    # the measured edge width (recommended). Pinning it
                                    # at 4 px cost -598 millipixels at 8 px of blur.
    halfwinK: float = 4.0           # window = halfwinK * RMS gradient width
    minContrast: float = 20.0       # skip lines below this peak-to-peak, 8-bit DN
    nFrames: int = 20               # frames averaged per reported measurement
    maxResidualFrac: float = MAX_RESID_FRAC   # refuse above this fraction of the
                                              # measured span. A rough sawn edge
                                              # sits near 0.1%; a fit spanning two
                                              # different edges reaches 100%.
    edgeTrim: float = 0.08          # fraction of each scan axis dropped at BOTH
                                    # ends before fitting. Real specimens have
                                    # rounded or chamfered corners, and near a
                                    # corner the outermost point sits on the arc
                                    # rather than the flat edge -- the points
                                    # curve away and drag the line. Unlike the
                                    # abrupt jump that tilt causes, this is
                                    # gradual, so valid_band cannot see it.
                                    # Trimming costs a little averaging and
                                    # removes the whole problem.
    trimCorners: bool = True        # drop scan lines that clip a corner instead of
                                    # crossing both end faces (see valid_band)
    autoOrient: bool = True         # report the larger dimension as LENGTH, so the
                                    # answer does not depend on which way round the
                                    # coupon was placed on the platform

    # ---- DERIVED, mirroring the CSV formulas --------------------------------
    @property
    def couponW(self) -> float:
        """Coupon width that actually fits -- derived from the panel, not chosen."""
        return (self.specW - (self.couponN - 1) * self.couponGap) / self.couponN

    @property
    def zSpec(self) -> float:
        """SPECIMEN PLANE -- the primary measurement datum."""
        return self.zPanBot + self.panT + self.ribH

    @property
    def fovL(self) -> float:
        return self.specL * self.fovMargin

    @property
    def fovW(self) -> float:
        """Short axis FOV -- set by the sensor aspect ratio, not by the specimen."""
        return self.fovL * self.sensorH / self.sensorW

    @property
    def mag(self) -> float:
        return self.sensorW / self.fovL

    @property
    def pxPitch(self) -> float:
        return self.sensorW / self.pixelsL

    @property
    def objDist(self) -> float:
        """Principal plane -> specimen PLANE. Equals workDist + lensPP."""
        return self.lensF * (1 + 1 / self.mag)

    @property
    def workDist(self) -> float:
        """Specimen plane to the front of the lens."""
        return self.objDist - self.lensPP

    @property
    def mmPerPxNominal(self) -> float:
        """Scale at the specimen PLANE. The scale actually used sits one specT higher."""
        return self.fovL / self.pixelsL

    @property
    def armZ(self) -> float:
        """Camera arm underside -- derived, so moving the specimen plane moves the camera."""
        return self.zSpec + self.workDist + self.camStack

    @property
    def focalPxNominal(self) -> float:
        """What --calibrate should land near, from first principles."""
        return self.lensF / self.pxPitch

    def sObj(self, thickness: Optional[float] = None) -> float:
        """Principal plane -> the silhouette-forming TOP face of the coupon."""
        t = self.specT if thickness is None else thickness
        s = self.objDist - t
        if s <= self.lensF:
            raise ValueError(f"nonsensical standoff: sObj = {s:.3f} mm")
        return s

    @property
    def effDistance(self) -> float:
        """A = objDist - lensF. The whole optical chain reduces to this one number.

        Thin lens: m = lensF/(s - lensF), so mm/px = pxPitch/m = (s - lensF)/focalPx.
        Substituting s = objDist - t gives mm/px = (A - t)/focalPx, with A absorbing
        both the standoff and the focal length. Two unknowns, A and focalPx, and
        --calib-stack solves for both from images alone."""
        return self.effDist if self.effDist > 0 else self.objDist - self.lensF

    def mmPerPx(self, thickness: Optional[float] = None) -> float:
        if self.focalPx <= 0:
            raise RuntimeError("focalPx not set - run --calibrate or --calib-stack")
        t = self.specT if thickness is None else thickness
        A = self.effDistance
        if A - t <= 0:
            raise ValueError(f"nonsensical geometry: A - t = {A - t:.3f} mm")
        return (A - t) / self.focalPx


# ============================================================== camera ======

# Every one of these is a CONTENT-DEPENDENT transform: it changes pixel values as a
# function of what is in the scene. That means the measured edge position moves when
# the scene moves, which is exactly the failure a metrology rig cannot tolerate. For
# measurement you want the sensor to behave like a boring linear photon counter.
#
#   awb / awb_gain / wb_mode  white balance -- per-channel gains that chase the scene
#   aec / aec2 / ae_level     auto exposure -- brightness changes between frames
#   agc / agc_gain            auto gain -- also raises read noise; pin it at minimum
#   bpc / wpc                 dead-pixel correction -- silently rewrites pixel values
#   raw_gma                   gamma -- OFF keeps the response linear, which is what the
#                             gradient-centroid estimator assumes
#   lenc                      lens shading correction -- a spatially varying gain, so
#                             the same edge reads differently at different field
#                             positions. Poison for this application.
#   dcw                       downsize/rescale -- an extra resampling stage
#   sharpness / denoise       edge enhancement. This one will physically MOVE your
#                             edges, and is the single worst setting to leave on.
CAM = [("awb", 0), ("awb_gain", 0), ("wb_mode", 0), ("aec", 0), ("aec2", 0),
       ("ae_level", 0), ("agc", 0), ("agc_gain", 0), ("gainceiling", 0),
       ("bpc", 0), ("wpc", 0), ("raw_gma", 0), ("lenc", 0), ("dcw", 0),
       ("special_effect", 0), ("hmirror", 0), ("vflip", 0),
       ("sharpness", 0), ("denoise", 0)]


def setup(cfg: Config) -> None:
    """Push the sensor into a fixed, deterministic state via the CameraWebServer
    /control endpoint. Some vars are sensor-specific and will 404 on an OV2640 --
    that is harmless, hence the warn-and-continue."""
    url = f"http://{cfg.host}/control"
    for var, val in [("framesize", cfg.framesize), ("quality", cfg.jpegQuality),
                     ("aec_value", cfg.exposure), *CAM]:
        try:
            requests.get(url, params={"var": var, "val": val}, timeout=3)
        except requests.RequestException as e:
            print(f"  [warn] {var}={val}: {e}")
    time.sleep(0.5)   # let the sensor settle on the new exposure before grabbing


def frame(cfg: Config) -> np.ndarray:
    """Grab one still and return it as float32 greyscale.

    float32, not uint8: the gradient and centroid arithmetic downstream needs
    fractional values, and doing it in integers would quantise away the subpixel
    information the whole method depends on."""
    r = requests.get(f"http://{cfg.host}/capture", timeout=10)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("JPEG decode failed")
    return img.astype(np.float32)


def load(path: str) -> np.ndarray:
    """Read an image FILE as float32 -- same output as frame(), so every downstream
    stage is identical whether the pixels came from a camera or from disk.

    Handles 16-bit input (a raw green plane from pi_capture.py) as well as 8-bit.
    Deeper data is rescaled to the 0-255 range as FLOAT, not as integers, so the
    extra bits survive into the edge estimator while minContrast and the DN
    diagnostics keep the same meaning across both sources."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"could not read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 255:
        # 12-bit raw sits in a 16-bit container; scale by the real range used,
        # not by 65535, or a 12-bit frame would come out 16x too dark.
        bits = 16 if img.max() > 4095 else (12 if img.max() > 1023 else 10)
        img *= 255.0 / (2 ** bits - 1)
    return img


def _peaks(prof: np.ndarray, k: int, sep: int) -> list[tuple[int, float]]:
    """Top k peaks by greedy non-maximum suppression. Dependency-free."""
    p = prof.copy().astype(np.float64)
    out = []
    for _ in range(k):
        i = int(np.argmax(p))
        if p[i] <= 0:
            break
        out.append((i, float(p[i])))
        p[max(0, i - sep):i + sep + 1] = 0
    return out


def auto_roi(img: np.ndarray, margin: int = 40, min_prominence: float = 3.0,
             min_contrast_hint: float = 20.0) -> tuple[int, int, int, int]:
    """Locate the specimen and return an ROI around it, from gradient projections.

    WHY NOT A THRESHOLD. Otsu picks one global level for the whole frame, which fails
    outright on an uneven backlight: on a real test frame the background was 217 DN
    above the coupon and 146 DN below it while the coupon sat at 76 DN, so the dim
    lower background landed on the coupon side of the split and merged with it. Every
    global threshold has this failure.

    WHAT THIS DOES INSTEAD. It looks for the four EDGES, not for the region. A slow
    illumination gradient has a small derivative spread over hundreds of pixels; a
    specimen edge has a large derivative concentrated in a few. Projecting the signed
    x-derivative down the rows gives sharp peaks on a flat baseline however uneven the
    lighting is: FALLING peaks (bright to dark) and RISING peaks (dark to bright).

    WHY PAIRS, NOT THE STRONGEST PEAK. A real rig has other full-height edges, and
    they can easily be stronger than the specimen. A backlight aperture inside a dark
    surround is the common case, and on a test frame its edges scored 24267 and 16574
    against the coupon's 13287. But polarity disambiguates them completely:

        dark surround -> bright aperture -> dark surround   gives RISE then FALL
        bright field  -> dark specimen   -> bright field    gives FALL then RISE

    So the specimen is the strongest fall/rise pair with the FALL on the left. Taking
    the global argmax of each profile instead picks one edge from the aperture and one
    from the coupon, which is what the earlier version did. Scoring on min() of the
    two rather than the sum insists both edges are real, so one very strong edge
    cannot drag in a weak partner.

    Small features are suppressed for free: pencil marks on a test coupon are strong
    gradients but span a few dozen rows out of hundreds, so they contribute a few
    percent of what a full-height edge does.

    `min_prominence` is peak height over the profile median. Below it the detection is
    untrustworthy and this raises rather than returning a plausible-looking box.
    """
    H, W = img.shape
    sm = cv2.GaussianBlur(img, (0, 0), 2.0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)

    def locate(g: np.ndarray, axis: int, n: int) -> tuple[int, int, float]:
        fall = np.clip(-g, 0, None).sum(axis=axis)   # bright -> dark
        rise = np.clip(g, 0, None).sum(axis=axis)    # dark -> bright
        edge = max(3, n // 50)                       # ignore frame-border artefacts
        fall[:edge] = fall[n - edge:] = 0
        rise[:edge] = rise[n - edge:] = 0

        # Two different jobs, previously conflated. `nms` is how far apart two
        # peaks must be to count as separate humps -- a few pixels, set by the
        # edge width. `pair` is the minimum extent of the object itself, and
        # scaling THAT to the frame was a bug: a thin specimen viewed edge-on is
        # far narrower than n/20, so its own edge pair was rejected as too close
        # together and a background feature won instead. On the side camera,
        # where the thickness is the small dimension, that is the normal case.
        nms = max(3, n // 200)
        pair = max(3, n // 400)
        F, R = _peaks(fall, 6, nms), _peaks(rise, 6, nms)
        base = max(float(np.median(fall[edge:n - edge])),
                   float(np.median(rise[edge:n - edge])), 1e-6)

        best = None
        for xf, sf in F:                             # specimen: FALL then RISE
            for xr, sr in R:
                if xr - xf < pair:
                    continue
                score = min(sf, sr)                  # both edges must be real
                if best is None or score > best[2]:
                    best = (xf, xr, score)

        if best is not None:
            # Take the OUTER foot of each gradient hump, not its peak. A tilted
            # specimen spreads one edge over L*sin(theta) rows, so its projection
            # is a broad plateau rather than a spike and the peak sits somewhere
            # in the middle of it -- which cropped a 636 px tall bar to 381 px at
            # 4 degrees and starved the perpendicular scan of lines. Walking out
            # to where the hump falls to a fifth of its height recovers the true
            # extent, and for an untilted specimen the hump is a spike so this
            # changes nothing.
            xf, xr, sc = best
            f_lo, r_hi = xf, xr
            while f_lo > edge and fall[f_lo - 1] > 0.2 * fall[xf]:
                f_lo -= 1
            while r_hi < n - edge - 1 and rise[r_hi + 1] > 0.2 * rise[xr]:
                r_hi += 1
            best = (f_lo, r_hi, sc)

        if best is None:
            raise RuntimeError(
                "no falling-then-rising edge pair found. Either no specimen is in "
                "frame, or it is BRIGHT on a dark background -- invert the image, "
                "the row scan expects dark on bright.")
        return best[0], best[1], best[2] / base

    x0, x1, px = locate(gx, 0, W)
    y0, y1, py = locate(gy, 1, H)
    if min(px, py) < min_prominence:
        # Name the axis. They fail for different reasons: a weak X usually means
        # something else in frame -- a platform edge, a bright strip -- is raising
        # the baseline the specimen has to stand out from; a weak Y often just
        # means the specimen presents a short edge to that axis after rotation.
        axis = "X (left/right edges)" if px < py else "Y (top/bottom edges)"
        raise RuntimeError(
            f"edge peaks too weak on {axis}: prominence X {px:.1f}, Y {py:.1f}, "
            f"need {min_prominence}. Either the specimen is low contrast, or "
            f"another feature in frame is competing. Set an explicit ROI, or "
            f"lower the prominence threshold.")

    box = (int(max(0, x0 - margin)), int(max(0, y0 - margin)),
           int(min(W, x1 + margin)), int(min(H, y1 + margin)))

    # Confirm the box actually contains a DARK OBJECT. The projections above find
    # strong intensity transitions, and a backlit specimen is only one thing that
    # produces those -- a platform rim, a bright strip, the edge of an aperture all
    # qualify. Lowering min_prominence to force a detection makes this more likely,
    # not less, and the failure then surfaces much later as "0 valid lines" from a
    # box sitting on empty background.
    bx0, by0, bx1, by1 = box
    inner = img[by0:by1, bx0:bx1]
    if inner.size:
        pad = max(10, margin)
        ox0, oy0 = max(0, bx0 - pad), max(0, by0 - pad)
        ox1, oy1 = min(W, bx1 + pad), min(H, by1 + pad)
        outer = img[oy0:oy1, ox0:ox1]
        dark = float(np.percentile(inner, 10))       # the specimen, if present
        bright = float(np.percentile(outer, 90))     # the lit background
        if bright - dark < 3 * min_contrast_hint:
            raise RuntimeError(
                f"found a box at {box} but it holds no dark object "
                f"(10th pct inside {dark:.0f}, 90th pct around {bright:.0f}). "
                f"Auto-ROI has locked onto a background feature -- a platform rim "
                f"or a bright strip. Set an explicit --roi rather than lowering "
                f"the prominence, which makes this more likely.")
    return box


# ====================================================== subpixel edges ======

def _row(prof: np.ndarray, cfg: Config) -> Optional[tuple[float, float]]:
    """Locate the left and right edge of one dark-on-bright intensity profile,
    to a fraction of a pixel.

    Two stages:

    1. COARSE -- find where the profile drops below the midpoint of its own min and
       max. Using a level derived per-line, rather than a fixed global threshold,
       makes this immune to uneven backlight illumination across the frame. This only
       needs to be right to a pixel or two; it just tells stage 2 where to look.

    2. REFINE -- gradient centroid:   x = sum(x * |g|) / sum(|g|)

       The intensity step is blurred into a ramp several pixels wide by diffraction,
       defocus and the pixel aperture. That blur is symmetric about the true edge, so
       the gradient |g| is a symmetric bump centred on it, and the centroid of a
       symmetric bump is its centre -- regardless of how wide the blur is, how bright
       the lamp is, or how much contrast there happens to be. That indifference is the
       whole point: a fixed threshold drifts as the lamp ages and as the specimen
       darkens, and the centroid does not.

       Two iterations: the first centroid is computed about the coarse estimate, which
       may sit a pixel or so off, making the window asymmetric and pulling the answer.
       Recentring on that result and recomputing removes almost all of it.
    """
    lo, hi = float(prof.min()), float(prof.max())
    if hi - lo < cfg.minContrast:
        return None                                # no specimen on this line
    idx = np.flatnonzero(prof < 0.5 * (lo + hi))   # coarse: below the 50% level
    if idx.size < 3:
        return None                                # too thin to be real

    g = np.gradient(prof)   # central differences -> no half-pixel shift, unlike np.diff

    def refine(x0: int) -> Optional[float]:
        x = float(x0)
        for _ in range(2):
            a, b = int(round(x)) - cfg.halfwin, int(round(x)) + cfg.halfwin + 1
            if a < 0 or b > prof.size:
                return None                        # edge too near the ROI border
            w = np.abs(g[a:b])                     # |gradient|: sign-blind, so the same
            if w.sum() <= 1e-9:                    # code handles both edges
                return None
            x = float(np.dot(np.arange(a, b), w) / w.sum())
        return x

    xl, xr = refine(int(idx[0])), refine(int(idx[-1]))   # first and last crossing
    return (xl, xr) if xl is not None and xr is not None and xr > xl else None


def grad_width(img: np.ndarray, cfg: Config, n: int = 16) -> Optional[float]:
    """RMS width of the gradient bump, in px, median over n sampled lines.

    This is the actual optical transition width -- diffraction plus defocus plus the
    pixel aperture plus our own pre-smoothing. The centroid window has to be scaled to
    it: a window narrower than the bump clips the tails, and because the clipping is
    asymmetric near a boundary the centroid walks inward. Measuring it beats guessing,
    because it changes whenever the lens, aperture or focus changes."""
    out = []
    for y in range(0, img.shape[0], max(1, img.shape[0] // n)):
        prof = img[y]
        lo, hi = float(prof.min()), float(prof.max())
        if hi - lo < cfg.minContrast:
            continue
        idx = np.flatnonzero(prof < 0.5 * (lo + hi))
        if idx.size < 3:
            continue
        g = np.abs(np.gradient(prof))
        for x0 in (int(idx[0]), int(idx[-1])):
            a, b = x0 - 20, x0 + 21
            if a < 0 or b > prof.size:
                continue
            w, xs = g[a:b], np.arange(a, b)
            if w.sum() <= 1e-9:
                continue
            mu = np.dot(xs, w) / w.sum()
            out.append(np.sqrt(max(np.dot((xs - mu) ** 2, w) / w.sum(), 0.0)))
    return float(np.median(out)) if out else None


def scan(roi: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply _row to every horizontal line of the ROI.

    Returns three parallel arrays: (row index, left edge x, right edge x), containing
    only the lines where detection succeeded. Rows that fail are dropped rather than
    filled, so a partial obstruction costs coverage but never corrupts the fit."""
    img = cv2.GaussianBlur(roi, (0, 0), cfg.sigma) if cfg.sigma > 0 else roi

    # Size the centroid window once per image from the measured edge width. Clamped
    # below so it never degenerates, and above so it cannot reach the opposite edge.
    if cfg.halfwin <= 0:
        gw = grad_width(img, cfg)
        cfg = replace(cfg, halfwin=int(np.clip(round(cfg.halfwinK * gw), 3, 30))
                      if gw else 6)

    out = [(y, *e) for y in range(img.shape[0]) if (e := _row(img[y], cfg))]
    if len(out) < 20:
        raise RuntimeError(f"only {len(out)} valid lines - check ROI, focus, exposure")
    a = np.array(out, float)
    return a[:, 0], a[:, 1], a[:, 2]


def valid_band(pos, lo, hi):
    """Rows where BOTH detected edges lie on the specimen's END faces.

    THE PROBLEM. Each scan line reports its outermost dark pixels. On a coupon
    square to the frame those are always the two end faces -- one straight edge
    each, and the line fit is clean. Tilt it and that stops being true: past the
    corner, the outermost point jumps onto a LONG edge and races sideways. The
    point set becomes two segments meeting at a corner, and one straight line
    forced through both gives a meaningless slope and an enormous residual.
    Measured on a 3400x400 px bar: at 0.5 deg the RMS is 0.49 px, at 2 deg it is
    329 px and the reported tilt runs to 68 degrees.

    THE TEST. The two regimes differ by orders of magnitude in how fast the edge
    moves per row. Along an end face the step is tan(theta) -- hundredths of a
    pixel. Along a long edge it is 1/tan(theta) -- tens of pixels. So the valid
    band is simply the longest run of rows over which the edge advances slowly,
    and that test is indifferent to taper, which changes the step by a fraction
    of a percent rather than a factor of a thousand.

    Short interruptions are bridged so that a nick or a dust speck splits nothing.

    LIMIT. A band where both ends are simultaneously on end faces exists only
    while tan(theta) < T/L -- about 6.7 degrees for a 8.5:1 coupon, and it
    narrows to nothing there. Beyond that no scan line crosses both end faces and
    the dimension is not recoverable this way; the caller gets an empty band and
    should say so rather than report a number.
    """
    def run_of(v):
        d = np.abs(np.diff(v))
        med = float(np.median(d)) if d.size else 0.0
        thr = max(1.0, 20.0 * med)          # adaptive: 20x the typical step
        ok = np.concatenate([[True], d < thr])
        best = (0, 0)
        i = 0
        while i < ok.size:
            if not ok[i]:
                i += 1
                continue
            j = i
            gap = 0
            while j + 1 < ok.size and (ok[j + 1] or gap < 3):
                gap = 0 if ok[j + 1] else gap + 1
                j += 1
            j -= gap
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j + 1
        return best

    a1, b1 = run_of(lo)
    a2, b2 = run_of(hi)
    a, b = max(a1, a2), min(b1, b2)         # both edges must be valid together
    return a, b


# ============================================================ line fit ======

class Line(NamedTuple):
    """An edge modelled as  x = slope * y + icept.

    Solved in this form rather than y = mx + c because a near-vertical edge has an
    infinite slope in the usual parameterisation."""
    slope: float
    icept: float
    rms: float      # residual RMS in px -- how straight the edge actually is
    n: int          # rows surviving outlier rejection


def fit(y: np.ndarray, x: np.ndarray, k: float = 3.0) -> Line:
    """Least-squares line through the subpixel edge points, with outlier rejection.

    WHY FIT A LINE AT ALL. A single row places its edge to maybe 0.1 px. Fitting a
    line through N of them averages that down as 1/sqrt(N) -- roughly 0.004 px on the
    intercept for 600 rows. That improvement is the entire reason this rig can beat
    its own pixel pitch, and it works only because the edge really is straight. The
    `rms` field is the check: if it climbs, the specimen edge is not straight and the
    averaging argument no longer holds.

    OUTLIER REJECTION uses the median absolute deviation rather than the standard
    deviation, because a single bad row (dust, a nick in the specimen, a JPEG ringing
    artefact) inflates sigma enough to hide itself. The 1.4826 converts MAD to an
    equivalent sigma for Gaussian noise. Three passes; it converges in one or two."""
    keep = np.ones(y.size, bool)
    m = c = 0.0
    for _ in range(3):
        if keep.sum() < 10:
            break
        m, c = np.linalg.lstsq(np.column_stack([y[keep], np.ones(keep.sum())]),
                               x[keep], rcond=None)[0]
        r = x - (m * y + c)                                   # residuals, ALL points
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        # Floor the scale. A near-perfect edge gives MAD ~ 0, and a 3-sigma gate around
        # zero rejects EVERY point -- the rms then averages an empty array and returns
        # nan. Synthetic data hits this immediately; noisy real data hides it until it
        # doesn't. 1 millipixel is far below anything the estimator can resolve.
        s = max(s, 1e-3)
        new = np.abs(r) < k * s
        # Never accept a pass that discards most of the data: that is a sign the model
        # is wrong (a curved edge, two objects in the ROI), not that the data is dirty.
        if new.sum() < max(10, 0.5 * keep.sum()):
            break
        if np.array_equal(new, keep):
            break                                             # converged
        keep = new
    r = x[keep] - (m * y[keep] + c)
    return Line(float(m), float(c), float(np.sqrt(np.mean(r ** 2))), int(keep.sum()))


def gap(a: Line, b: Line, at: float = 0.0) -> float:
    """Perpendicular distance between two nominally parallel edges.

    Both lines are x = m*y + c. `at` is the position along the scan axis where the
    separation is evaluated, and it MATTERS the moment the two edges are not
    parallel: the intercept difference is the separation extrapolated to position
    0, which is one END of the ROI, not the middle. On a coupon that tapers by 10%
    that reported the narrow end and was 5% low against the true mean -- a real
    error found on real specimens, not a hypothetical. Pass the midpoint of the
    scanned range and you get the width at the centre, which for parallel edges is
    identical and for a tapered one is representative.

    Note that a single number cannot describe a coupon whose width genuinely
    varies; use chords.py for the profile. This makes the single number the least
    misleading one available. But if the specimen is rotated in the frame by
    theta = atan(m), that horizontal gap is the true width divided by cos(theta) -- you
    are measuring a diagonal slice across the part. Hence:

        true width = |dc| * cos(atan(m)) = |dc| / sqrt(1 + m^2)

    Small angle, real cost: at just 1.6 degrees of tilt the uncorrected value is 5 um
    high on a 12.5 mm width. Averaging the two slopes assumes the edges are parallel,
    which for a machined coupon they are to well within the noise."""
    m = 0.5 * (a.slope + b.slope)
    sep = abs((b.slope * at + b.icept) - (a.slope * at + a.icept))
    return sep / np.sqrt(1 + m * m)


# ========================================================= measurement ======

class Result(NamedTuple):
    width_mm: float
    length_mm: float
    width_px: float
    length_px: float
    um_per_px: float
    tilt_deg: float         # specimen rotation in frame; diagnostic, already corrected
    edge_rms_px: float      # edge straightness; watch this, it flags a bad setup early


def _check_fit(a: Line, b: Line, span: float, what: str,
               frac: float = MAX_RESID_FRAC) -> None:
    """Refuse to return a number when the straight-line model has clearly failed.

    The residual RMS separates the two regimes by orders of magnitude. A real
    sawn wood edge sits around 1.7 px -- that is genuine fibre tear-out, and the
    line still averages it down to a few microns. A fit spanning two different
    edges of a rotated coupon reads 200-500 px. Anything approaching a percent of
    the dimension being measured is not a rough edge, it is the wrong model, and
    the millimetres it produces look entirely plausible.
    """
    lim = max(3.0, frac * span)
    worst = max(a.rms, b.rms)
    if worst > lim:
        raise RuntimeError(
            f"{what}: edge fit residual {worst:.1f} px ({worst/span*100:.1f}% of "
            f"the {span:.0f} px span) - the straight-line model does not describe "
            f"these edges. Too much rotation, or the edge genuinely is not "
            f"straight. Raise --max-residual to see the number anyway, and run "
            f"chords.py on a captured frame to see the edge's actual shape.")


def measure(img: np.ndarray, cfg: Config, thickness: Optional[float] = None) -> Result:
    """One frame in, dimensions out."""
    x0, y0, x1, y1 = cfg.roi
    H, W = img.shape
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise ValueError(f"roi {cfg.roi} does not fit inside a {W}x{H} frame")
    roi = img[y0:y1, x0:x1]

    # WIDTH: scan rows, giving a left and right edge per row.
    y, xl, xr = scan(roi, cfg)
    if cfg.trimCorners:
        a, b = valid_band(y, xl, xr)
        if b - a < 20:
            raise RuntimeError(
                f"only {b - a} rows cross both end faces - the specimen is too "
                f"rotated for this axis. Square it to the frame.")
        y, xl, xr = y[a:b + 1], xl[a:b + 1], xr[a:b + 1]
    if cfg.edgeTrim > 0 and y.size > 40:
        sp = y.max() - y.min()
        k = (y >= y.min() + cfg.edgeTrim * sp) & (y <= y.max() - cfg.edgeTrim * sp)
        if k.sum() > 20:
            y, xl, xr = y[k], xl[k], xr[k]
    left, right = fit(y, xl), fit(y, xr)
    w_px = gap(left, right, float(np.median(y)))
    _check_fit(left, right, w_px, "rows", cfg.maxResidualFrac)

    # LENGTH: identical problem rotated 90 degrees, so transpose and reuse everything.
    # ascontiguousarray because .T only flips strides, and the row-wise slicing in
    # scan() is far slower on a non-contiguous view.
    ty, tl, tr = scan(np.ascontiguousarray(roi.T), cfg)
    if cfg.trimCorners:
        a, b = valid_band(ty, tl, tr)
        if b - a < 20:
            raise RuntimeError(
                f"only {b-a} columns cross both end faces - too rotated.")
        ty, tl, tr = ty[a:b + 1], tl[a:b + 1], tr[a:b + 1]
    if cfg.edgeTrim > 0 and ty.size > 40:
        sp = ty.max() - ty.min()
        k = (ty >= ty.min() + cfg.edgeTrim * sp) & (ty <= ty.max() - cfg.edgeTrim * sp)
        if k.sum() > 20:
            ty, tl, tr = ty[k], tl[k], tr[k]
    top, bot = fit(ty, tl), fit(ty, tr)
    l_px = gap(top, bot, float(np.median(ty)))
    _check_fit(top, bot, l_px, "columns", cfg.maxResidualFrac)

    # PIXELS -> MILLIMETRES. See the module docstring for why thickness subtracts.
    # PIXELS -> MILLIMETRES. Thin lens, not pinhole: see the module docstring for why
    # the lensF term matters (321 ppm at specTden, 1655 ppm at specTraw).
    mmpx = cfg.mmPerPx(thickness)

    # The row scan gives the HORIZONTAL extent and the column scan the VERTICAL one.
    # Which is "length" depends only on how the operator laid the coupon down, so
    # assign by magnitude: on a tensile coupon the length is always the larger of the
    # two (couponW 21.25 mm vs specL 200 mm). Set cfg.autoOrient = False to keep the
    # raw image axes.
    if cfg.autoOrient and w_px > l_px:
        w_px, l_px = l_px, w_px

    return Result(w_px * mmpx, l_px * mmpx, w_px, l_px, mmpx * 1000,
                  float(np.degrees(np.arctan(0.5 * (left.slope + right.slope)))),
                  0.5 * (left.rms + right.rms))


def run(cfg: Config, thickness: Optional[float]) -> None:
    """Measure repeatedly and report mean and spread."""
    w, l, r = [], [], None
    for i in range(cfg.nFrames):
        try:
            r = measure(frame(cfg), cfg, thickness)
        except Exception as e:
            print(f"  {i:2d}: FAILED - {e}")     # keep going; one bad frame is not fatal
            continue
        w.append(r.width_mm)
        l.append(r.length_mm)
        print(f"  {i:2d}:  W {r.width_mm:8.4f}  L {r.length_mm:8.4f} mm   "
              f"tilt {r.tilt_deg:+5.2f} deg   edge RMS {r.edge_rms_px:.3f} px")

    if len(w) < 2:
        return print("\nnot enough frames for statistics")
    w, l = np.array(w), np.array(l)
    print(f"\n  n = {len(w)}/{cfg.nFrames}   scale {r.um_per_px:.4f} um/px")
    print(f"  WIDTH   {w.mean():8.4f} mm   1s {w.std(ddof=1)*1000:6.1f} um")
    print(f"  LENGTH  {l.mean():8.4f} mm   1s {l.std(ddof=1)*1000:6.1f} um")
    # The spread is REPEATABILITY, not accuracy. It tells you nothing about whether the
    # mean is right. Averaging drives the random term down as 1/sqrt(N) and does
    # precisely nothing to a systematic error -- a wrong thickness, a moved lens or a
    # stale calibration will give you a beautifully precise wrong answer.
    print("  (repeatability only - averaging does nothing to bias)")


def calibrate(known_mm: float, height_mm: float, cfg: Config) -> None:
    """Solve focalPx from an artefact of known width at a known height above zSpec:

        focalPx = width_px * (s_cal - lensF) / known_mm

    Calibration still needs one length standard even though nothing sits in the frame
    during measurement. Use something dimensionally stable and certified -- a slip
    gauge, a ground pin, a precision ring -- not a ruler or a printed target.

    height_mm sets s_cal exactly the way specT sets sObj, so an error here becomes a
    fixed bias on every later measurement. Calibrate at the geometry you will measure
    at and the standoff cancels in the ratio (see module docstring)."""
    s_cal = cfg.effDistance - height_mm      # A - h, the only distance that matters
    cfg.focalPx = 1.0                 # placeholder: measure() refuses on 0, and only
                                      # width_px is wanted here, so the scale is moot
    px = []
    for i in range(cfg.nFrames):
        try:
            px.append(measure(frame(cfg), cfg, height_mm).width_px)
        except Exception as e:
            print(f"  {i:2d}: FAILED - {e}")
    if len(px) < 2:
        raise RuntimeError("calibration failed")
    a_ = np.array(px)
    f = a_.mean() * s_cal / known_mm
    print(f"\n  A - h    {s_cal:.3f} mm  (A {cfg.effDistance:.3f} - height {height_mm})")
    # The ppm figure is your noise floor: no later measurement can beat it.
    print(f"  measured {a_.mean():.3f} px  (1s {a_.std(ddof=1):.3f} px, "
          f"{a_.std(ddof=1)/a_.mean()*1e6:.0f} ppm)")
    print(f"  focalPx  = {f:.2f}   ->  {s_cal/f*1000:.4f} um/px")
    dev = (f/cfg.focalPxNominal - 1) * 100
    print(f"  nominal  = {cfg.focalPxNominal:.2f} px (lensF/pxPitch)   deviation {dev:+.2f}%")
    if abs(dev) > 5:
        print("  >> more than 5% off nominal: the built geometry is not what the CAD says,")
        print("     or lensF / lensPP / the datum stack is wrong. Investigate before trusting.")
    print(f"\n  Put into Config:  focalPx = {f:.2f}   calibSObj = {s_cal:.3f}")


def calib_stack(pairs: list[tuple[str, float]], known_mm: float,
                cfg: Config) -> tuple[float, float]:
    """Solve BOTH optical unknowns from images of one artefact at several heights.

    The scale law is  mm/px = (A - h) / focalPx, where h is the height of the
    silhouette-forming face above the platform and A = objDist - lensF. For a fixed
    artefact of width W0 imaged at height h:

        W0 = px(h) * (A - h) / focalPx      ->      px(h) * (A - h) = W0 * focalPx

    The right side is constant, so px*(A-h) is the same at every height. With two
    heights that is two equations in two unknowns:

        A       = (px1*h1 - px2*h2) / (px1 - px2)
        focalPx = px1 * (A - h1) / W0

    Why this matters more than it looks: it calibrates the SLOPE of the thickness
    correction, not just the scale. On this geometry the correction is 12210 ppm at
    specTden, and getting its slope wrong by a few percent is the error that
    calibration at a single height cannot see. It also means you never need lensF,
    lensPP or the principal plane location -- which on an ESP32-CAM module you have
    no way to look up.

    Use three or more heights: the fit residual is then a real check that the model
    holds, rather than an assumption. Heights must be KNOWN -- stack slip gauges."""
    if len(pairs) < 2:
        raise ValueError("need at least two heights")

    obs = []
    for path, h in pairs:
        img = load(path)
        c = replace(cfg, roi=auto_roi(img), focalPx=1.0, effDist=1e6)
        px = measure(img, c, 0.0).width_px
        obs.append((h, px))
        print(f"    h = {h:7.3f} mm   {px:10.4f} px")

    h = np.array([o[0] for o in obs], float)
    px = np.array([o[1] for o in obs], float)

    # px*(A - h) = C  ->  px*A - C = px*h  ->  linear in (A, C) with px*h as the target
    (A, C), *_ = np.linalg.lstsq(np.column_stack([px, -np.ones(px.size)]),
                                 px * h, rcond=None)
    focalPx = C / known_mm

    pred = C / (A - h)                       # predicted px at each height
    resid_ppm = (px - pred) / px * 1e6
    print(f"\n  A       = {A:10.3f} mm   (platform -> equivalent pinhole, lensF folded in)")
    print(f"  focalPx = {focalPx:10.2f} px")
    print(f"  scale at h=0: {A / focalPx * 1000:.4f} um/px")
    print(f"  fit residuals: " + "  ".join(f"{r:+.0f}" for r in resid_ppm) + " ppm")
    worst = np.max(np.abs(resid_ppm))
    print(f"  worst {worst:.0f} ppm -- {'good' if worst < 500 else 'SUSPECT: check the'
          + ' heights, or the artefact moved between shots'}")
    print(f"\n  Put into Config:  effDist = {A:.3f}   focalPx = {focalPx:.2f}")
    return A, focalPx


CALIB_FILE = "calibration.json"


def save_calib(path: str, focalPx: float, cfg: Config, **prov) -> None:
    """Persist the calibration so it survives between runs.

    A calibration is a property of the OPTICS, not of any specimen, so retyping it
    on every command is both tedious and a live source of error -- a mistyped digit
    scales every result and nothing in the output looks wrong. Provenance is stored
    alongside it because a calibration you cannot date or trace is one you cannot
    decide whether to trust."""
    rec = {"focalPx": round(focalPx, 4), "effDist": round(cfg.effDistance, 4),
           "when": datetime.now(timezone.utc).isoformat(timespec="seconds"), **prov}
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"  saved to {path}")


def scale_focal(focalPx: float, calib_width: int, image_width: int) -> float:
    """Rescale focalPx when the image resolution differs from the calibration.

    focalPx is a length expressed IN PIXELS, so it is meaningless without the
    resolution that produced it. Calibrating on a 4056 px frame and then
    measuring a 2028 px green plane -- which is exactly what happens when you
    move from a processed JPEG to raw Bayer -- halves every pixel count, and
    without this the result comes out 2x wrong while looking entirely plausible.
    """
    if not calib_width or calib_width == image_width:
        return focalPx
    return focalPx * image_width / calib_width


def load_calib(path: str, cfg: Config) -> bool:
    """Apply a stored calibration. Returns True if one was applied.

    Loudly, never silently: a measurement whose scale came from an unseen file is
    exactly the kind of result that gets trusted when it should not be."""
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    cfg.focalPx = float(rec["focalPx"])
    stored = float(rec["effDist"])
    print(f"  calibration from {path}: focalPx {cfg.focalPx:.2f}, "
          f"effDist {stored:.1f} mm, {rec.get('when', 'undated')}")
    # effDist is part of the calibration geometry. If the camera has since moved,
    # the stored focalPx no longer describes this setup.
    if cfg.effDist > 0 and abs(cfg.effDist - stored) > 0.5:
        print(f"  >> WARNING: --eff-dist {cfg.effDist:.1f} differs from the "
              f"calibrated {stored:.1f} mm.")
        print("     If the camera really moved, recalibrate. If you just mistyped it,")
        print("     fix the number -- the scale is wrong either way.")
    else:
        cfg.effDist = stored
    return True


LOG_FIELDS = ["when", "image", "length_mm", "width_mm", "length_px", "width_px",
              "um_per_px", "tilt_deg", "edge_rms_px", "specT_mm", "effDist_mm",
              "focalPx", "roi", "note"]


def log_row(path: str, cfg: Config, r: "Result", image: str, note: str = "") -> None:
    """Append one measurement to a CSV, creating it with a header if new.

    The calibration context is written alongside every row on purpose. A number
    like "199.98 mm" is not interpretable six months later without knowing which
    focalPx and effDist produced it and what thickness was assumed -- and those
    are exactly the things that change when someone nudges the rig. Storing them
    per row means a later recalibration can be applied retrospectively, and a
    drifted session can be identified rather than guessed at.
    """
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "when":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image":       image,
            "length_mm":   f"{r.length_mm:.4f}",
            "width_mm":    f"{r.width_mm:.4f}",
            "length_px":   f"{r.length_px:.4f}",
            "width_px":    f"{r.width_px:.4f}",
            "um_per_px":   f"{r.um_per_px:.4f}",
            "tilt_deg":    f"{r.tilt_deg:+.4f}",
            "edge_rms_px": f"{r.edge_rms_px:.4f}",
            "specT_mm":    f"{cfg.specT:.4f}",
            "effDist_mm":  f"{cfg.effDistance:.3f}",
            "focalPx":     f"{cfg.focalPx:.4f}",
            "roi":         "|".join(str(v) for v in cfg.roi),
            "note":        note,
        })


def diagnose(img: np.ndarray, cfg: Config) -> None:
    """Report WHY rows are being rejected, instead of just how many.

    `scan` deliberately drops any line it cannot measure cleanly, which is right for
    a measurement but useless for debugging: "0 valid lines" has at least five
    distinct causes and they need opposite fixes. This replays the same logic and
    counts each one."""
    x0, y0, x1, y1 = cfg.roi
    H, W = img.shape
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise SystemExit(f"roi {cfg.roi} does not fit inside {W}x{H}")
    raw = img[y0:y1, x0:x1]
    roi = cv2.GaussianBlur(raw, (0, 0), cfg.sigma) if cfg.sigma > 0 else raw
    print(f"ROI {cfg.roi} -> {roi.shape[1]} x {roi.shape[0]} px")
    print(f"  levels: min {raw.min():.0f}  max {raw.max():.0f}  "
          f"5th {np.percentile(raw,5):.0f}  95th {np.percentile(raw,95):.0f}")

    gw = grad_width(roi, cfg)
    hw = cfg.halfwin if cfg.halfwin > 0 else (
        int(np.clip(round(cfg.halfwinK * gw), 3, 30)) if gw else 6)
    print(f"  edge width {gw if gw else float('nan'):.2f} px RMS -> window +/-{hw} px\n")

    reasons = {"ok": 0, "low contrast": 0, "<3 dark px": 0,
               "window off left": 0, "window off right": 0, "xr <= xl": 0}
    for y in range(roi.shape[0]):
        pr = roi[y]
        lo, hi = float(pr.min()), float(pr.max())
        if hi - lo < cfg.minContrast:
            reasons["low contrast"] += 1; continue
        idx = np.flatnonzero(pr < 0.5 * (lo + hi))
        if idx.size < 3:
            reasons["<3 dark px"] += 1; continue
        if idx[0] - hw < 0:
            reasons["window off left"] += 1; continue
        if idx[-1] + hw + 1 > pr.size:
            reasons["window off right"] += 1; continue
        if idx[-1] <= idx[0]:
            reasons["xr <= xl"] += 1; continue
        reasons["ok"] += 1
    print("  rows, by outcome:")
    for k, v in reasons.items():
        if v:
            print(f"    {k:<18}{v:5d}")

    mid = roi[roi.shape[0] // 2]
    lo, hi = float(mid.min()), float(mid.max())
    idx = np.flatnonzero(mid < 0.5 * (lo + hi))
    print(f"\n  middle row: min {lo:.0f} max {hi:.0f}, 50% level {(lo+hi)/2:.0f}")
    if idx.size:
        print(f"    first dark px at x={idx[0]}, last at x={idx[-1]} "
              f"(ROI is 0..{mid.size-1})")
        print(f"    -> needs {hw} px clear each side; has {idx[0]} left, "
              f"{mid.size-1-idx[-1]} right")
    step = max(1, mid.size // 60)
    print("    profile: " + "".join(
        "#" if v < (lo+hi)/2 else "." for v in mid[::step]))
    print(f"             {'^ left edge of ROI':<30}{'right edge ^':>30}")


def overlay(path: str, cfg: Config) -> None:
    """Dump an annotated frame. Look at this before trusting any number.

    Check: the ROI box contains the specimen and nothing else; the detected edge dots
    run cleanly down both sides with no gaps or excursions; the row count is close to
    the full ROI height. Also check the histogram -- the bright field should sit near
    200 DN, not clipped at 255."""
    img = frame(cfg)
    x0, y0, x1, y1 = cfg.roi
    y, xl, xr = scan(img[y0:y1, x0:x1], cfg)
    vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 180, 255), 1)
    for yy, a, b in zip(y, xl, xr):
        cv2.circle(vis, (int(a) + x0, int(yy) + y0), 0, (0, 0, 255), -1)
        cv2.circle(vis, (int(b) + x0, int(yy) + y0), 0, (0, 255, 0), -1)
    cv2.imwrite(path, vis)
    L, R = fit(y, xl), fit(y, xr)
    print(f"{path}: {L.n}/{y1-y0} rows, edge RMS {L.rms:.3f}/{R.rms:.3f} px")


def geometry(cfg: Config) -> None:
    """Print the derived chain. Cross-check these against Onshape before building."""
    print("  INPUTS")
    for k in ("specL","specW","specTraw","specTden","couponN","couponGap",
              "zPanBot","panT","ribH","zBacklight","fovMargin","sensorW","sensorH",
              "pixelsL","lensF","lensPP","camStack"):
        print(f"    {k:<12}{getattr(cfg,k)}")
    print("  DERIVED")
    for k in ("couponW","zSpec","fovL","fovW","mag","workDist","armZ",
              "objDist","pxPitch","mmPerPxNominal","focalPxNominal"):
        v = getattr(cfg, k)
        print(f"    {k:<16}{v:12.6f}")
    print(f"    {'mmPerPx@specT':<16}{(cfg.sObj(cfg.specTden)-cfg.lensF)/cfg.focalPxNominal:12.6f}"
          f"   (um/px: {(cfg.sObj(cfg.specTden)-cfg.lensF)/cfg.focalPxNominal*1000:.3f})")
    print("  BUDGET")
    for lbl, dim, tgt in (("length", cfg.specL, 25.0), ("width", cfg.couponW, 20.0)):
        print(f"    {lbl:<8}{tgt:5.1f} um on {dim:7.2f} mm = {tgt/1000/dim*1e6:6.0f} ppm")
    print(f"    specT sensitivity: {1/(cfg.objDist-cfg.lensF)*1e6:.0f} ppm per mm of "
          f"thickness error")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host")
    p.add_argument("--thickness", type=float, help="specT: measured coupon thickness, mm")
    p.add_argument("--calibrate", type=float, metavar="MM",
                   help="known size of the artefact, mm")
    p.add_argument("--calib-file", default=CALIB_FILE,
                   help=f"where the calibration is stored (default {CALIB_FILE})")
    p.add_argument("--calib-on", choices=("length", "width"), default="length",
                   help="which dimension --calibrate refers to. Auto-orient reports "
                        "the LARGER dimension as length, so a known overall size is "
                        "almost always the length. Default: length.")
    p.add_argument("--calib-stack", type=float, metavar="MM",
                   help="artefact width; pass image:height pairs as positional args")
    p.add_argument("pairs", nargs="*", metavar="IMAGE:HEIGHT_MM")
    p.add_argument("--halfwin", type=int, metavar="PX",
                   help="force the centroid window instead of sizing it from the "
                        "measured edge width. A window narrower than the gradient "
                        "bump biases INWARD -- we measured -598 millipixels at 8 px "
                        "of blur with halfwin=4. Use only to get a rough number out "
                        "of a badly defocused frame, and treat it as ~1%% accurate.")
    p.add_argument("--eff-dist", type=float, metavar="MM",
                   help="A = camera-to-PLATE distance (lensF folded in). On a rig "
                        "with unknown optics, your measured standoff is a good "
                        "approximation: the principal plane and lensF together are "
                        "usually a few mm, i.e. ~1-2%% at a 200 mm standoff.")
    p.add_argument("--focal-px", type=float, metavar="PX", help="skip calibration")
    p.add_argument("--no-auto-orient", action="store_true",
                   help="keep raw image axes instead of calling the larger LENGTH")
    p.add_argument("--log", metavar="CSV",
                   help="append the measurement, and the calibration that produced "
                        "it, to this CSV")
    p.add_argument("--note", help="free-text note stored with the logged row")
    p.add_argument("--diagnose", action="store_true",
                   help="report why rows are rejected, instead of measuring")
    p.add_argument("--geometry", action="store_true", help="print the derived chain and exit")
    p.add_argument("--image", metavar="PATH", help="measure a saved photo instead of the camera")
    p.add_argument("--auto-roi", action="store_true", help="locate the specimen automatically")
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1",
                   help="set the ROI explicitly, overriding Config and --auto-roi")
    p.add_argument("--overlay", metavar="PATH", help="save an annotated diagnostic frame")
    p.add_argument("--frames", type=int)
    p.add_argument("--no-setup", action="store_true", help="skip sensor configuration")
    a = p.parse_args()

    cfg = Config()
    if a.host:
        cfg.host = a.host
    if a.frames:
        cfg.nFrames = a.frames
    if a.thickness is not None:
        cfg.specT = a.thickness
    if a.no_auto_orient:
        cfg.autoOrient = False
    if a.roi:
        try:
            v = [int(t) for t in a.roi.split(",")]
            if len(v) != 4:
                raise ValueError
        except ValueError:
            raise SystemExit("--roi wants four integers: X0,Y0,X1,Y1")
        cfg.roi = tuple(v)
    if a.halfwin:
        cfg.halfwin = a.halfwin
    if a.eff_dist:
        cfg.effDist = a.eff_dist
    if a.focal_px:
        cfg.focalPx = a.focal_px
    elif not a.calibrate and not a.calib_stack:
        load_calib(a.calib_file, cfg)
    if a.geometry:
        return geometry(cfg)
    if a.calib_stack:
        pairs = [(p_.rsplit(":", 1)[0], float(p_.rsplit(":", 1)[1])) for p_ in a.pairs]
        if len(pairs) < 2:
            raise SystemExit("need >=2 IMAGE:HEIGHT_MM pairs, e.g. h0.jpg:0 h1.jpg:10.02")
        return calib_stack(pairs, a.calib_stack, cfg) and None

    # ---- offline mode: one saved photo, no camera, no repeats ----
    if a.image:
        img = load(a.image)
        if a.auto_roi and not a.roi:          # an explicit --roi always wins
            cfg.roi = auto_roi(img)
            print(f"auto ROI {cfg.roi}  (image {img.shape[1]}x{img.shape[0]})")
        else:
            print(f"ROI {cfg.roi}  (image {img.shape[1]}x{img.shape[0]})")
        if a.calibrate:
            if a.thickness is None:
                raise SystemExit("--calibrate needs --thickness")
            s_cal = cfg.effDistance - (a.thickness or 0.0)
            cfg.focalPx = 1.0
            r0 = measure(img, cfg, a.thickness)
            ref = r0.length_px if a.calib_on == "length" else r0.width_px
            oth = r0.width_px if a.calib_on == "length" else r0.length_px
            f = ref * s_cal / a.calibrate
            print(f"  {a.calib_on} {ref:.4f} px across {a.calibrate} mm, "
                  f"A-t = {s_cal:.3f} mm")
            print(f"  focalPx = {f:.2f}   ->  {s_cal/f*1000:.4f} um/px")
            fov = 2*np.degrees(np.arctan((img.shape[1]/2)/f))
            print(f"  implied horizontal field of view: {fov:.0f} deg", end="")
            print("   <-- IMPLAUSIBLE, check --eff-dist and --calib-on"
                  if not 20 < fov < 120 else "   (plausible)")
            other = "width" if a.calib_on == "length" else "length"
            print(f"  cross-check: {other.upper()} {oth:.4f} px -> {oth*s_cal/f:.4f} mm")
            print(f"     compare that against your calipers. Agreement means the")
            print(f"     pipeline is sound; disagreement is lens distortion, which on")
            print(f"     a short ESP32 lens can easily be 1-3%.")
            save_calib(a.calib_file, f, cfg, image=a.image, known_mm=a.calibrate,
                       axis=a.calib_on, thickness=a.thickness,
                       imageWidth=int(img.shape[1]))
            print(f"  subsequent runs pick this up automatically; "
                  f"--focal-px still overrides it")
            return
        if a.diagnose:
            print("\n--- rows (width scan) ---")
            diagnose(img, cfg)
            print("\n--- columns (length scan) ---")
            c2 = replace(cfg, roi=(cfg.roi[1], cfg.roi[0], cfg.roi[3], cfg.roi[2]))
            diagnose(np.ascontiguousarray(img.T), c2)
            return
        r = measure(img, cfg, a.thickness)
        print(f"  WIDTH   {r.width_px:9.4f} px   {r.width_mm:9.4f} mm")
        print(f"  LENGTH  {r.length_px:9.4f} px   {r.length_mm:9.4f} mm")
        print(f"  scale {r.um_per_px:.4f} um/px   tilt {r.tilt_deg:+.3f} deg   "
              f"edge RMS {r.edge_rms_px:.4f} px")
        if a.log:
            log_row(a.log, cfg, r, a.image, a.note or "")
            print(f"  logged to {a.log}")
        return

    if not a.no_setup:
        setup(cfg)
    if a.overlay:
        return overlay(a.overlay, cfg)
    if a.calibrate:
        if a.thickness is None:
            raise SystemExit("--calibrate needs --thickness (artefact height above platform)")
        return calibrate(a.calibrate, a.thickness, cfg)
    run(cfg, a.thickness)


if __name__ == "__main__":
    main()