#!/usr/bin/env python3
"""
pi_dual.py -- two HQ cameras: bottom for length and width, side for thickness.

    python3 pi_dual.py --eff-dist0 775 --eff-dist1 300
    python3 pi_dual.py --eff-dist0 775 --eff-dist1 300 --log runs.csv

    desktop      http://<pi>:8080
    touchscreen  http://<pi>:8080/touch
"""

import argparse
import collections
import csv
import http.server
import json
import os
import socket
import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2

import chord_measure as C
import gauge as G


# ------------------------------------------------------------------ camera --

class Cam:
    """Hardware only: one sensor, its streams, its preview buffer.
    """

    def __init__(self, index, exposure, size=(4056, 3040)):
        self.index = index
        self.lock = threading.Lock()
        self.jpeg = None
        self.pc = Picamera2(index)
        self.pc.configure(self.pc.create_video_configuration(
            main={"size": size, "format": "RGB888"},
            # 64-aligned, so the ISP adds no stride padding for the preview to
            # misread as picture.
            lores={"size": (512, 384), "format": "YUV420"},
            buffer_count=2,
        ))
        if exposure:
            self.pc.set_controls({"AeEnable": False, "ExposureTime": exposure})
        self.pc.start()

    def grab(self):
        with self.lock:
            rgb = self.pc.capture_array("main")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def preview(self):
        with self.lock:
            yuv = self.pc.capture_array("lores")
        return cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR)

    def close(self):
        try:
            self.pc.stop(); self.pc.close()
        except Exception:
            pass


class Role:
    """A measurement job -- bottom or side -- with its own standoff, ROI and
    calibration, pointed at whichever Cam currently performs it.
    """

    def __init__(self, name, eff_dist, roi, prominence, calib_file, max_resid):
        self.name, self.calib_file = name, calib_file
        self.cam = None
        self.cfg = G.Config()
        self.cfg.effDist = eff_dist
        self.cfg.maxResidualFrac = max_resid
        self.prominence = prominence
        self.calib_w = 0
        self.focalPx = 0.0
        self.last_roi = None
        self.last_w = 0
        if G.load_calib(calib_file, self.cfg):
            self.focalPx = self.cfg.focalPx
            try:
                self.calib_w = int(json.load(open(calib_file)).get("imageWidth", 0))
            except Exception:
                pass
            if eff_dist:
                self.cfg.effDist = eff_dist
        else:
            print(f"# {name}: no {calib_file} -- reporting pixels")
        self.roi_spec = ([float(v) for v in roi.split(",")] if roi else None)
        self.roi_frac = (self.roi_spec is not None
                         and all(v <= 1.0 for v in self.roi_spec))

    def grab(self):
        return self.cam.grab()

    def find_roi(self, img):
        """Resolve the ROI and remember it for the preview overlay.
        """
        self.last_w = img.shape[1]
        try:
            if self.roi_spec is None:
                r = G.auto_roi(img, min_prominence=self.prominence)
            else:
                h, w = img.shape
                r = ((int(self.roi_spec[0] * w), int(self.roi_spec[1] * h),
                      int(self.roi_spec[2] * w), int(self.roi_spec[3] * h))
                     if self.roi_frac
                     else tuple(int(v) for v in self.roi_spec))
        except Exception:
            self.last_roi = None
            raise
        self.last_roi = r
        return r

    def configure_for(self, img, thickness):
        """The Config chords.py needs: this role's ROI and calibration, with
        focalPx rescaled to whatever resolution the frame actually is."""
        c = G.replace(self.cfg, roi=self.find_roi(img))
        c.focalPx = (G.scale_focal(self.focalPx, self.calib_w, img.shape[1])
                     if self.focalPx > 0 else 1.0)
        c.specT = thickness
        return c

    def measure(self, img, thickness):
        """`thickness` is the offset of this role's silhouette plane from its
        reference: specT for the bottom view, 0 for the side view, whose
        reference face IS its silhouette plane."""
        c = G.replace(self.cfg, roi=self.find_roi(img))
        c.focalPx = (G.scale_focal(self.focalPx, self.calib_w, img.shape[1])
                     if self.focalPx > 0 else 1.0)
        r = G.measure(img, c, thickness)
        k = c.mmPerPx(thickness) if self.focalPx > 0 else 0.0
        return r, c, k

    def calibrate(self, px, known_mm, axis, thickness, width):
        s_cal = self.cfg.effDistance - thickness
        if s_cal <= 0:
            raise ValueError("thickness exceeds eff-dist")
        f = float(px * s_cal / known_mm)
        self.focalPx, self.calib_w = f, int(width)
        G.save_calib(self.calib_file, f, G.replace(self.cfg, focalPx=f),
                     image=f"live-{self.name}", known_mm=known_mm, axis=axis,
                     thickness=thickness, imageWidth=int(width))
        fov = 2 * np.degrees(np.arctan((width / 2) / f))
        return (f"{self.name}: focalPx {f:.2f} | {s_cal/f*1000:.4f} um/px | "
                f"FOV {fov:.1f} deg"
                + ("" if 15 < fov < 100 else "  <-- IMPLAUSIBLE, check --eff-dist"))


# -------------------------------------------------------------------- html --

PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>dual</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{background:#1a1a1a;color:#e8e8e8;font:14px system-ui;margin:0;padding:16px}
.wrap{display:flex;gap:16px;flex-wrap:wrap}
.cam{flex:1;min-width:330px}.cam img{width:100%;border:1px solid #444}
.cap{font-size:12px;color:#888;margin:4px 0}
.panel{min-width:320px;flex:1}
.big{font:700 34px ui-monospace,monospace;color:#8fd9b0;margin:0}
.big small{font:400 15px system-ui;color:#777;margin-left:6px}
.lab{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:#7d7d7d;margin-top:12px}
.sub{font-size:11px;color:#888}
table{border-collapse:collapse;width:100%;margin-top:12px}
td{padding:4px 8px;border-bottom:1px solid #333}td:first-child{color:#999}
td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.bad{color:#e06c5a}.warn{color:#e8a33d}.ok{color:#6ec48f}
button{font:600 14px system-ui;padding:10px 18px;margin:10px 6px 0 0;border:0;
border-radius:6px;background:#2d7d5a;color:#fff}
input,select{padding:9px;background:#1e1e1e;color:#eee;border:1px solid #3a3a3a;
border-radius:6px}
#msg{color:#9c9;font-size:12px;min-height:34px;margin-top:8px}
</style></head><body>
<div class="wrap">
 <div class="cam"><img src="/stream0"><div class="cap" id="cap0">bottom - length &amp; width</div></div>
 <div class="cam"><img src="/stream1"><div class="cap" id="cap1">side - thickness &amp; length check</div></div>
 <div class="panel">
  <div class="lab">Length</div><div class="big" id="L">--</div>
  <div class="lab">Width</div><div class="big" id="W">--</div>
  <div class="lab">Thickness</div><div class="big" id="T">--</div>
  <div class="lab">Length cross-check</div><div class="big" id="D">--</div>
  <div class="sub" id="Ds">side camera vs bottom camera</div>
  <table id="q"></table>
  <div class="lab">Calibrate</div>
  <input id="km" type="number" step="0.001" placeholder="known mm" style="width:110px">
  <select id="cam" onchange="axes()"><option value="0">bottom</option>
   <option value="1">side</option></select>
  <select id="ax"><option value="length">length</option><option value="width">width</option></select>
  <button onclick="cal()">Calibrate</button>
  <button onclick="post('/log')">Log</button>
  <button onclick="post('/profile')">Profile</button>
  <button onclick="if(confirm('Swap the cameras? Both calibrations will need redoing.'))post('/swap')"
   style="background:#7d5a2d">Swap cameras</button>
  <div id="msg"></div>
  <pre id="pf" style="background:#111;padding:9px;border-radius:5px;font-size:11px;
   white-space:pre;overflow-x:auto;color:#bbb;margin-top:8px"></pre>
 </div>
</div><script>
function el(i){return document.getElementById(i)}
function say(t){var v=String(t||'');el('msg').textContent=v.length>200?v.slice(0,200):v}
function row(k,v,c){return '<tr><td>'+k+'</td><td class="'+(c||'')+'">'+v+'</td></tr>'}
function fmt(id,v,u,cal){el(id).innerHTML=(v===null||v===undefined)?'--':
 v.toFixed(cal?3:1)+'<small>'+u+'</small>'}
async function poll(){
 try{const s=await (await fetch('/state')).json();
  fmt('L',s.length_mm,s.cal0?'mm':'px',s.cal0);
  fmt('W',s.width_mm,s.cal0?'mm':'px',s.cal0);
  fmt('T',s.thickness_mm,s.cal1?'mm':'px',s.cal1);
  if(s.delta_mm===null||s.delta_mm===undefined){el('D').innerHTML='--'}
  else{el('D').innerHTML=(s.delta_mm>=0?'+':'')+s.delta_mm.toFixed(3)+
   '<small>mm</small>';el('D').style.color=Math.abs(s.delta_mm)<0.1?'#8fd9b0':'#e8a33d'}
  el('cap0').textContent='bottom (sensor '+s.sensor0+') - length & width';
  el('cap1').textContent='side (sensor '+s.sensor1+') - thickness & length check';
  el('q').innerHTML=
   row('bottom status',s.err0||'ok',s.err0?'bad':'ok')+
   row('side status',s.err1||'ok',s.err1?'bad':'ok')+
   row('bottom edge RMS',s.rms0===null?'--':s.rms0.toFixed(3)+' px')+
   row('side edge RMS',s.rms1===null?'--':s.rms1.toFixed(3)+' px')+
   row('bottom tilt',s.tilt0===null?'--':s.tilt0.toFixed(2)+' deg')+
   row('side tilt',s.tilt1===null?'--':s.tilt1.toFixed(2)+' deg')+
   row('pairing','bottom=sensor '+s.sensor0+', side=sensor '+s.sensor1,
       s.swapped?'warn':'')+
   row('plane',s.fixedPlane===null||s.fixedPlane===undefined?
       'floating (uses measured thickness)':'fixed at '+s.fixedPlane+' mm')+
   row('thickness 1s',s.tsd===null?'--':(s.tsd*1000).toFixed(1)+' um over '+s.n)+
   row('rate',s.rate.toFixed(2)+' Hz');
 }catch(e){}
 setTimeout(poll,700);
}
async function post(u){say('working...');
 const t=await (await fetch(u,{method:'POST'})).text();
 if(u==='/profile'){el('pf').textContent=t;say('profiled')}else{say(t)}}
// The side view's small axis is the THICKNESS, not a width. The value posted
// stays "width" because that is the scan axis the engine uses; only the label
// changes, so the operator is never asked to calibrate a dimension the specimen
// does not have from that viewpoint.
function axes(){var side=el('cam').value==='1';
 el('ax').options[1].text=side?'thickness':'width'}
async function cal(){const v=parseFloat(el('km').value);
 if(!(v>0)){say('enter the known size first');return}
 say('calibrating...');
 say(await (await fetch('/calibrate',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({known_mm:v,camera:parseInt(el('cam').value),
   axis:el('ax').value})})).text())}
axes();poll();
</script></body></html>"""

TOUCH = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>gauge</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>*{box-sizing:border-box;-webkit-user-select:none;user-select:none;
touch-action:manipulation}
html,body{margin:0;height:100%;background:#101010;color:#eee;font:16px system-ui;
overflow:hidden}
.pane{height:100%;display:flex;flex-direction:column;padding:16px;gap:6px}
.lab{font-size:17px;letter-spacing:.13em;color:#7d7d7d;text-transform:uppercase}
.val{font:700 66px ui-monospace,monospace;color:#7fd8a8;line-height:1.05;margin:0}
.val small{font:400 21px system-ui;color:#6a6a6a;margin-left:7px}
.chk{font:600 26px ui-monospace,monospace;color:#9a9a9a}
.spacer{flex:1}
input,select{min-height:66px;font-size:26px;background:#1e1e1e;color:#eee;
border:1px solid #3a3a3a;border-radius:11px;padding:0 12px;text-align:center}
.row{display:flex;gap:10px}.row>*{flex:1}
button{width:100%;min-height:76px;font:600 26px system-ui;border:0;border-radius:11px;
background:#2d7d5a;color:#fff;margin-top:9px}
button:active{background:#256b4c}
#msg{min-height:38px;font-size:15px;color:#8fbf9f;overflow:hidden}
.stale{color:#8a5a52 !important}
</style></head><body><div class="pane">
<div><div class="lab">Length</div><div class="val" id="L">--</div></div>
<div><div class="lab">Width</div><div class="val" id="W">--</div></div>
<div><div class="lab">Thickness</div><div class="val" id="T">--</div></div>
<div><div class="lab">Length check</div><div class="chk" id="D">--</div></div>
<div class="spacer"></div>
<div id="msg"></div>
<div class="row">
 <input id="km" type="number" step="0.001" placeholder="known mm">
 <select id="cam" onchange="axes()"><option value="0">bottom</option>
  <option value="1">side</option></select>
 <select id="ax"><option value="length">length</option><option value="width">width</option></select>
</div>
<button onclick="cal()">Calibrate</button>
<button onclick="post('/log')">Log reading</button>
<button onclick="post('/profile')">Profile</button>
<button onclick="if(confirm('Swap cameras? Recalibrate after.'))post('/swap')"
 style="background:#7d5a2d">Swap cameras</button>
</div><script>
function el(i){return document.getElementById(i)}
function say(t){var v=String(t||'');el('msg').textContent=v.length>150?v.slice(0,150):v}
function fmt(id,v,u,cal){var e=el(id);
 if(v===null||v===undefined){e.innerHTML='--';e.classList.add('stale')}
 else{e.innerHTML=v.toFixed(cal?2:1)+'<small>'+u+'</small>';e.classList.remove('stale')}}
async function poll(){
 try{const s=await (await fetch('/state')).json();
  fmt('L',s.length_mm,s.cal0?'mm':'px',s.cal0);
  fmt('W',s.width_mm,s.cal0?'mm':'px',s.cal0);
  fmt('T',s.thickness_mm,s.cal1?'mm':'px',s.cal1);
  el('D').textContent=(s.delta_mm===null||s.delta_mm===undefined)?'--':
   (s.delta_mm>=0?'+':'')+s.delta_mm.toFixed(3)+' mm';
  if(s.err0||s.err1)say((s.err0||'')+' '+(s.err1||''));
 }catch(e){}
 setTimeout(poll,700);
}
async function post(u){say('working...');
 const t=await (await fetch(u,{method:'POST'})).text();
 if(u==='/profile'){el('pf').textContent=t;say('profiled')}else{say(t)}}
// The side view's small axis is the THICKNESS, not a width. The value posted
// stays "width" because that is the scan axis the engine uses; only the label
// changes, so the operator is never asked to calibrate a dimension the specimen
// does not have from that viewpoint.
function axes(){var side=el('cam').value==='1';
 el('ax').options[1].text=side?'thickness':'width'}
async function cal(){const v=parseFloat(el('km').value);
 if(!(v>0)){say('enter the known size first');return}
 say('calibrating...');
 say(await (await fetch('/calibrate',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({known_mm:v,camera:parseInt(el('cam').value),
   axis:el('ax').value})})).text())}
poll();
</script></body></html>"""


LOG_COLS = ["when", "length_mm", "width_mm", "thickness_mm", "length_side_mm",
            "delta_mm", "length_px", "width_px", "thickness_px", "length_side_px",
            "tilt0_deg", "tilt1_deg", "rms0_px", "rms1_px",
            "focalPx0", "effDist0", "focalPx1", "effDist1", "roi0", "roi1", "note"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eff-dist0", type=float, default=775.0,
                   help="bottom camera: platform to lens, mm")
    p.add_argument("--eff-dist1", type=float, default=300.0,
                   help="side camera: coupon reference face to lens, mm")
    p.add_argument("--plane-offset", type=float, default=None, metavar="MM",
                   help="height of the bottom view's silhouette plane above its "
                        "reference, when that height is FIXED. Use 0 for the "
                        "chair geometry, where the coupon rests on glass and the "
                        "silhouette face is the glass plane. Omit it and the "
                        "bottom view instead takes the thickness measured by the "
                        "side view, and waits for it.")
    p.add_argument("--roi0"); p.add_argument("--roi1")
    p.add_argument("--exposure0", type=int); p.add_argument("--exposure1", type=int)
    p.add_argument("--prominence", type=float, default=3.0)
    p.add_argument("--max-residual", type=float, default=0.01)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--chords", type=int, default=20, help="chords per axis")
    p.add_argument("--band", type=float, default=3.0,
                   help="mm of specimen averaged into each chord; 0 = single row")
    p.add_argument("--trim", type=float, default=0.05,
                   help="fraction dropped at each end, where chords clip corners")
    p.add_argument("--order", type=int, default=3,
                   help="centreline polynomial order; 1 assumes straight edges")
    p.add_argument("--profile-csv", metavar="FILE",
                   help="Profile also writes every chord to this CSV")
    p.add_argument("--log", metavar="CSV"); p.add_argument("--note", default="")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--calib0", default="calib_bottom.json")
    p.add_argument("--calib1", default="calib_side.json")
    p.add_argument("--state", default="dual_state.json",
                   help="remembers which sensor is bottom and which is side")
    p.add_argument("--swap", action="store_true",
                   help="start with the two cameras exchanged")
    a = p.parse_args()

    cams = [Cam(0, a.exposure0), Cam(1, a.exposure1)]
    bottom = Role("bottom", a.eff_dist0, a.roi0, a.prominence,
                  a.calib0, a.max_residual)
    side = Role("side", a.eff_dist1, a.roi1, a.prominence,
                a.calib1, a.max_residual)

    # Which sensor performs which role is persisted, because a restart that
    # silently reverted the pairing would produce confident, wrong numbers: the
    # side view measured with the bottom view's standoff and calibration.
    def load_swap():
        try:
            return bool(json.load(open(a.state)).get("swapped", False))
        except Exception:
            return False

    swapped = {"v": a.swap or load_swap()}

    def bind():
        i, j = (1, 0) if swapped["v"] else (0, 1)
        bottom.cam, side.cam = cams[i], cams[j]

    def save_swap():
        try:
            json.dump({"swapped": swapped["v"]}, open(a.state, "w"))
        except Exception as e:
            print(f"# could not save {a.state}: {e}")

    bind()
    time.sleep(1.5)

    hist = collections.deque(maxlen=a.window)
    latest = {"thickness": None}
    snap = {"json": json.dumps({"rate": 0.0}).encode(), "row": None}

    def _row(r0, r1, c0, c1, specT, side_len, d):
        return {
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "length_mm": f"{d['length_mm']:.4f}",
            "width_mm": f"{d['width_mm']:.4f}",
            "thickness_mm": f"{specT:.4f}",
            "length_side_mm": f"{side_len:.4f}",
            "delta_mm": f"{d['delta_mm']:+.4f}",
            "length_px": f"{r0.length_px:.4f}",
            "width_px": f"{r0.width_px:.4f}",
            "thickness_px": f"{r1.width_px:.4f}",
            "length_side_px": f"{r1.length_px:.4f}",
            "tilt0_deg": f"{r0.tilt_deg:+.4f}",
            "tilt1_deg": f"{r1.tilt_deg:+.4f}",
            "rms0_px": f"{r0.edge_rms_px:.4f}",
            "rms1_px": f"{r1.edge_rms_px:.4f}",
            "focalPx0": f"{c0.focalPx:.4f}",
            "effDist0": f"{c0.effDistance:.3f}",
            "focalPx1": f"{c1.focalPx:.4f}",
            "effDist1": f"{c1.effDistance:.3f}",
            "roi0": "|".join(str(v) for v in c0.roi),
            "roi1": "|".join(str(v) for v in c1.roi),
            "note": a.note,
        }

    def publish(d, row=None):
        snap["json"] = json.dumps(d).encode()
        snap["row"] = row

    def preview_loop(cam):
        while True:
            try:
                bgr = cam.preview()
                # Whichever role is bound to this sensor right now -- so the
                # overlay follows a camera swap without restarting anything.
                role = bottom if bottom.cam is cam else side
                if role.last_roi and role.last_w:
                    k = bgr.shape[1] / role.last_w
                    x0, y0, x1, y1 = (int(v * k) for v in role.last_roi)
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 200, 255), 1)
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    cam.jpeg = buf.tobytes()
            except Exception:
                pass
            time.sleep(0.08)

    def measure_loop():
        while True:
            t0 = time.time()
            d = {"rate": 0.0, "err0": None, "err1": None,
                 "cal0": bottom.focalPx > 0, "cal1": side.focalPx > 0,
                 "swapped": swapped["v"],
                 "sensor0": bottom.cam.index, "sensor1": side.cam.index,
                 "length_mm": None, "width_mm": None, "thickness_mm": None,
                 "delta_mm": None, "tilt0": None, "tilt1": None,
                 "rms0": None, "rms1": None, "tsd": None, "n": len(hist)}
            row = None

            # --- SIDE FIRST: it produces the thickness the bottom camera needs.
            # Its own silhouette plane is the coupon's reference face, so it is
            # measured at offset 0 and needs nothing from the other camera.
            r1 = c1 = k1 = None
            try:
                img1 = side.grab()
                r1, c1, k1 = side.measure(img1, 0.0)
                d["tilt1"], d["rms1"] = float(r1.tilt_deg), float(r1.edge_rms_px)
                d["thickness_mm"] = float(r1.width_px * k1) if k1 else float(r1.width_px)
                if k1:
                    latest["thickness"] = float(r1.width_px * k1)
            except Exception as e:
                d["err1"] = str(e)[:90]

            if a.plane_offset is not None:
                specT = a.plane_offset
            else:
                specT = float(r1.width_px * k1) if (r1 is not None and k1) else None
            try:
                img0 = bottom.grab()
                r0, c0, _ = bottom.measure(img0, 0.0)
                d["tilt0"], d["rms0"] = float(r0.tilt_deg), float(r0.edge_rms_px)
                k0 = (c0.mmPerPx(specT)
                      if (bottom.focalPx > 0 and specT is not None) else 0.0)
                d["length_mm"] = float(r0.length_px * k0) if k0 else float(r0.length_px)
                d["width_mm"] = float(r0.width_px * k0) if k0 else float(r0.width_px)
                if k0 and k1:
                    side_len = float(r1.length_px * k1)
                    d["delta_mm"] = side_len - d["length_mm"]
                    hist.append(specT)
                    row = _row(r0, r1, c0, c1, specT, side_len, d)
            except Exception as e:
                d["err0"] = str(e)[:90]

            if d["err1"] and specT is None:
                d["err0"] = d["err0"] or "no thickness from the side view"
            d["fixedPlane"] = a.plane_offset

            if len(hist) > 1:
                arr = np.array(hist)
                d["tsd"] = float(arr.std(ddof=1))
            d["n"] = len(hist)
            d["rate"] = 1.0 / max(time.time() - t0, 1e-3)
            publish(d, row)
            if d["err0"] or d["err1"]:
                time.sleep(0.3)

    for c in cams:
        threading.Thread(target=preview_loop, args=(c,), daemon=True).start()
    threading.Thread(target=measure_loop, daemon=True).start()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, body, ctype="text/plain"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _mjpeg(self, get_cam):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    j = get_cam().jpeg
                    if j:
                        self.wfile.write(
                            b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/":
                self._send(PAGE.encode(), "text/html")
            elif self.path == "/touch":
                self._send(TOUCH.encode(), "text/html")
            elif self.path == "/state":
                self._send(snap["json"], "application/json")
            elif self.path == "/stream0":
                self._mjpeg(lambda: bottom.cam)     # resolved per frame, so a
            elif self.path == "/stream1":          # swap takes effect live
                self._mjpeg(lambda: side.cam)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/calibrate":
                n = int(self.headers.get("Content-Length", 0))
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                    known = float(req["known_mm"])
                    which = int(req.get("camera", 0))
                    axis = req.get("axis", "length")
                except Exception as e:
                    return self._send(f"bad request: {e}".encode())
                cam = bottom if which == 0 else side
                try:
                    img = cam.grab()
                    # Calibrate at the same offset the camera measures at: specT
                    # for the bottom camera comes from cam1; the side camera's
                    # reference face IS its silhouette plane, so it is zero.
                    if which == 0:
                        if a.plane_offset is not None:
                            # Fixed plane: no dependency, calibrate in any order.
                            t = a.plane_offset
                        elif latest["thickness"] is None:
                            return self._send(
                                b"calibrate the side view first: with a floating "
                                b"silhouette plane the bottom view's scale "
                                b"depends on the thickness, and calibrating at "
                                b"the wrong thickness biases every later length "
                                b"by (A-t)/A. Pass --plane-offset 0 if the coupon "
                                b"rests on glass, and the dependency disappears.")
                        else:
                            t = latest["thickness"]
                    else:
                        t = 0.0
                    save = cam.focalPx
                    cam.focalPx = 0.0
                    r, c, _ = cam.measure(img, t)
                    cam.focalPx = save
                    px = r.length_px if axis == "length" else r.width_px
                    return self._send(cam.calibrate(px, known, axis, t,
                                                    img.shape[1]).encode())
                except Exception as e:
                    return self._send(f"calibration failed: {e}".encode())

            if self.path == "/profile":
                out, rows = [], []
                if a.plane_offset is not None:
                    t, tnote = a.plane_offset, ""
                elif latest["thickness"] is not None:
                    t, tnote = latest["thickness"], ""
                else:
                    t, tnote = 0.0, ("  [no thickness available: bottom mm are "
                                     "scaled at offset 0]")
                jobs = [("side", side, 0.0, "thickness"),
                        ("bottom", bottom, t, "width")]
                if tnote:
                    out.append(tnote.strip())
                for label, role, off, small in jobs:
                    try:
                        img = role.grab()
                        c = role.configure_for(img, off)
                        band = a.band if role.focalPx > 0 else 0.0
                        lp, sp = C.measure_chords(img, c, off, a.chords, band,
                                                  a.trim, a.order)
                    except Exception as e:
                        out.append(f"{label}: {str(e)[:80]}")
                        continue
                    cal = role.focalPx > 0
                    u = "mm" if cal else "px"
                    for pr, nm in ((lp, "length"), (sp, small)):
                        v = (pr.spans() if cal
                             else np.array([ch.span_px for ch in pr.chords]))
                        rng = float(v.max() - v.min())
                        out.append(
                            f"{label} {nm}: mean {v.mean():.4f} {u}  "
                            f"min {v.min():.4f}  max {v.max():.4f}  "
                            f"range {rng*1000:.1f} um" if cal else
                            f"{label} {nm}: mean {v.mean():.3f} px  "
                            f"range {rng:.3f} px")
                        if cal:
                            out.append("  " + C.sparkline(pr))
                        if pr.shape:
                            k = pr.mmpx * 1000 if cal else 1.0
                            uu = "um" if cal else "px"
                            for sd, sh in (("A", pr.shape["lo"]),
                                           ("B", pr.shape["hi"])):
                                out.append(f"  edge {sd}: wave "
                                           f"{sh['wave_ptp']*k:.1f} {uu} p-p, "
                                           f"rough {sh['rough_rms']*k:.1f} {uu} rms")
                        for i, ch in enumerate(pr.chords):
                            rows.append([label, nm, i, f"{ch.pos_mm:.4f}",
                                         f"{ch.span_px:.4f}", f"{ch.span_mm:.4f}",
                                         ch.n_rows, f"{ch.scatter_px:.4f}",
                                         f"{pr.tilt_deg:.4f}"])
                if a.profile_csv and rows:
                    new = (not os.path.exists(a.profile_csv)
                           or os.path.getsize(a.profile_csv) == 0)
                    with open(a.profile_csv, "a", newline="") as fh:
                        w = csv.writer(fh)
                        if new:
                            w.writerow(["view", "axis", "index", "pos_mm", "span_px",
                                        "span_mm", "n_rows", "scatter_px", "tilt_deg"])
                        w.writerows(rows)
                    out.append(f"wrote {len(rows)} chords to {a.profile_csv}")
                return self._send("\n".join(out).encode())

            if self.path == "/swap":
                swapped["v"] = not swapped["v"]
                bind()
                save_swap()
                # The thickness history describes the old pairing; keeping it
                # would blend two different geometries into one spread figure.
                hist.clear()
                latest["thickness"] = None
                return self._send(
                    (f"swapped: bottom is now sensor "
                     f"{bottom.cam.index}, side is sensor {side.cam.index}. "
                     f"Recalibrate both -- the calibrations belong to the "
                     f"viewpoints, and the viewpoints have changed sensors."
                     ).encode())

            if self.path == "/log":
                if not a.log:
                    return self._send(b"no --log file given")
                row = snap["row"]
                if not row:
                    return self._send(b"no complete measurement to log")
                new = not os.path.exists(a.log) or os.path.getsize(a.log) == 0
                with open(a.log, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=LOG_COLS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                return self._send(f"logged to {a.log}".encode())
            self.send_error(404)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()

    print(f"\n  bottom = sensor {bottom.cam.index}  focalPx {bottom.focalPx:.2f}"
          f"  A {bottom.cfg.effDistance:.1f} mm")
    print(f"  side   = sensor {side.cam.index}  focalPx {side.focalPx:.2f}"
          f"  A {side.cfg.effDistance:.1f} mm")
    if a.plane_offset is not None:
        print(f"  FIXED PLANE at {a.plane_offset:.3f} mm -- cameras independent, "
              f"thickness is reported, not required")
    else:
        print("  floating plane -- the bottom view waits on the side view's "
              "thickness")
    print(f"\n  desktop      http://{ip}:{a.port}")
    print(f"  touchscreen  http://{ip}:{a.port}/touch\n")
    try:
        http.server.ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams:
            c.close()


if __name__ == "__main__":
    main()