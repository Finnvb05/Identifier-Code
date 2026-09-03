#!/usr/bin/env python3
"""
pi_measure.py -- live view that continuously measures what the camera sees.

    python3 pi_measure.py --thickness 5.5
    python3 pi_measure.py --thickness 5.5 --raw --log runs.csv

Reads calibration.json, so calibrate first with gauge.py. Open the printed URL.

The preview streams at video rate; the measurement runs as fast as it can in the
background, typically around 1 Hz at full resolution. That is deliberate -- the
edge fit across ~1500 rows is where the accuracy comes from, and subsampling
rows to hit a higher frame rate would trade away the thing being measured.

The rolling spread is the number worth watching. A single reading tells you
almost nothing; the 1-sigma over the last N frames of an undisturbed coupon is
your repeatability, and it is the first figure that says what the rig is worth.
"""

import argparse
import collections
import http.server
import json
import socket
import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2

import chord_measure as C
import gauge as G
from pi_capture import green_plane

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>measure</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>body{background:#1a1a1a;color:#e8e8e8;font:14px system-ui;margin:0;padding:16px;
-webkit-user-select:none;user-select:none;touch-action:manipulation}
.wrap{display:flex;gap:18px;flex-wrap:wrap}img{border:1px solid #444;max-width:760px;width:100%}
.p{min-width:300px;flex:1}.big{font:700 30px ui-monospace,monospace;color:#8fd9b0;margin:2px 0}
.u{font:400 14px system-ui;color:#888}.sub{color:#999;font-size:12px;margin-bottom:12px}
table{border-collapse:collapse;width:100%;margin-top:10px}
td{padding:3px 8px;border-bottom:1px solid #333}td:first-child{color:#999}
td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.bad{color:#e06c5a}.warn{color:#e8a33d}.ok{color:#6ec48f}
button{font:600 15px system-ui;padding:10px 22px;margin-top:12px;border:0;border-radius:6px;
background:#2d7d5a;color:#fff;cursor:pointer}button:hover{background:#379a6e}
h3{margin:16px 0 2px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#777}
#m{color:#9c9;min-height:20px;font-size:12px}
/* Small screens: a measurement station needs the NUMBERS and the buttons, not a
   video feed. The preview collapses to a thumbnail that can be tapped away
   entirely, the readouts grow, and every control clears the ~44 px minimum a
   fingertip needs -- the desktop buttons are about 38 px and get mis-tapped. */
@media (max-width:820px){
  body{padding:8px;font-size:15px}
  .wrap{flex-direction:column;gap:8px}
  img{max-width:100%;max-height:22vh;object-fit:contain}
  img.off{display:none}
  .big{font-size:40px;margin:0}
  h3{margin:8px 0 0;font-size:11px}
  .sub{font-size:11px;margin-bottom:4px}
  td{padding:5px 8px}
  button{padding:14px 18px;font-size:16px;min-height:48px;margin:6px 4px 0 0}
  input,select{min-height:48px;font-size:16px}
  #pf{font-size:10px}
}
#hide{display:none}
@media (max-width:820px){#hide{display:inline-block;background:#444}}
</style></head><body>
<div class="wrap"><div><img src="/stream"></div><div class="p">
<h3>Length</h3><div class="big" id="L">--<span class="u"> mm</span></div>
<div class="sub" id="Ls"></div>
<h3>Width</h3><div class="big" id="W">--<span class="u"> mm</span></div>
<div class="sub" id="Ws"></div>
<h3>Quality</h3><table id="q"></table>
<h3>Calibrate from this view</h3>
<div class="sub">Point at an artefact of known size, wait for the spread to settle,
then enter its true size.</div>
<input id="km" type="number" step="0.001" placeholder="known mm" style="width:110px;
 padding:8px;background:#222;border:1px solid #444;color:#eee;border-radius:5px">
<select id="ax" style="padding:8px;background:#222;border:1px solid #444;color:#eee;
 border-radius:5px"><option value="length">length</option><option value="width">width</option></select>
<button onclick="cal()">Calibrate</button>
<button onclick="log()">Log reading</button>
<button onclick="prof()">Profile</button>
<button id="hide" onclick="var v=document.getElementById('vid');
 v.classList.toggle('off')">Preview</button><div id="m"></div>
<pre id="pf" style="background:#111;padding:10px;border-radius:5px;font-size:11px;
 white-space:pre;overflow-x:auto;color:#bbb;margin-top:10px"></pre>
</div></div><script>
function r(k,v,c){return '<tr><td>'+k+'</td><td class="'+(c||'')+'">'+v+'</td></tr>'}
async function poll(){try{const s=await (await fetch('/state')).json();
 if(s.ok){
  const U=s.calibrated?' mm':' px', f=s.calibrated?3:2;
  const Lv=s.calibrated?s.length_mm:s.length_px, Wv=s.calibrated?s.width_mm:s.width_px;
  const Ls=s.calibrated?s.Lsd*1000:s.Lsd_px, Us=s.calibrated?' um':' px';
  document.getElementById('L').innerHTML=Lv.toFixed(f)+'<span class="u">'+U+'</span>';
  document.getElementById('W').innerHTML=Wv.toFixed(f)+'<span class="u">'+U+'</span>';
  document.getElementById('Ls').textContent=s.n>1?('1s '+Ls.toFixed(f===3?1:3)+Us+
    '  over '+s.n+' frames'+(s.calibrated?'':'  -- not calibrated')):'collecting...';
  document.getElementById('Ws').textContent='';
  document.getElementById('q').innerHTML=
    r('edge RMS',s.edge_rms.toFixed(4)+' px',s.edge_rms<0.1?'ok':'warn')+
    r('tilt',s.tilt.toFixed(3)+' deg')+
    (s.calibrated?r('scale',s.um_per_px.toFixed(4)+' um/px'):r('scale','not calibrated','warn'))+
    r('clipped',s.clipped.toFixed(2)+'%',s.clipped>0.01?'bad':'ok')+
    r('bright field',s.p95+' / '+s.full)+
    r('rate',s.rate.toFixed(2)+' Hz')+
    r('frame',s.frame)+r('ROI',s.roi);
 } else { document.getElementById('q').innerHTML=r('status',s.err,'bad');
   document.getElementById('L').innerHTML='--';
   document.getElementById('W').innerHTML='--';
   document.getElementById('Ls').textContent=''; }
}catch(e){}setTimeout(poll,600)}
async function log(){document.getElementById('m').textContent=
 await (await fetch('/log',{method:'POST'})).text()}
async function prof(){const e=document.getElementById('pf');
 e.textContent='profiling...';
 e.textContent=await (await fetch('/profile',{method:'POST'})).text()}
async function cal(){const v=parseFloat(document.getElementById('km').value);
 if(!(v>0)){document.getElementById('m').textContent='enter the known size first';return}
 document.getElementById('m').textContent='calibrating...';
 const r=await fetch('/calibrate',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({known_mm:v,axis:document.getElementById('ax').value})});
 document.getElementById('m').textContent=await r.text()}
poll();</script></body></html>"""


# A separate page for the operator touchscreen. Deliberately NOT the desktop
# layout scaled down: it carries only the four numbers that matter at the bench
# and the three controls, sized for a fingertip on a 720x1560 portrait panel.
# Everything else -- preview, clipping, rows, rate -- stays on the desktop view,
# where there is room for it and someone is looking for it.
TOUCH_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>gauge</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;touch-action:manipulation}
html,body{margin:0;height:100%;background:#101010;color:#eee;
 font:16px system-ui,sans-serif;overflow:hidden}
.pane{height:100%;display:flex;flex-direction:column;padding:18px;gap:10px}
.lab{font-size:20px;letter-spacing:.14em;color:#7d7d7d;text-transform:uppercase}
.val{font:700 92px ui-monospace,SFMono-Regular,monospace;color:#7fd8a8;
 line-height:1;margin:2px 0 0;overflow:hidden;text-overflow:clip}
.val small{font:400 26px system-ui;color:#6a6a6a;margin-left:8px}
.two{display:flex;gap:14px;margin-top:2px}
.two div{flex:1;background:#1b1b1b;border-radius:10px;padding:12px 14px}
.two .lab{font-size:15px}
.two .n{font:600 30px ui-monospace,monospace;color:#d8d8d8;margin-top:4px}
.spacer{flex:1}
.row{display:flex;gap:12px}
input,select{flex:1;min-height:76px;font-size:30px;background:#1e1e1e;color:#eee;
 border:1px solid #3a3a3a;border-radius:12px;padding:0 16px;text-align:center}
button{width:100%;min-height:88px;font:600 30px system-ui;border:0;border-radius:12px;
 background:#2d7d5a;color:#fff;margin-top:12px}
button:active{background:#256b4c}
#msg{min-height:44px;font-size:17px;color:#8fbf9f;line-height:1.3;
 overflow:hidden;margin-top:6px}
.stale{color:#8a5a52 !important}
</style></head><body><div class="pane">
<div><div class="lab">Length</div><div class="val" id="L">--</div></div>
<div><div class="lab">Width</div><div class="val" id="W">--</div></div>
<div class="two">
 <div><div class="lab">Tilt</div><div class="n" id="T">--</div></div>
 <div><div class="lab">Scale</div><div class="n" id="S">--</div></div>
</div>
<div class="spacer"></div>
<div id="msg"></div>
<div class="row">
 <input id="km" type="number" step="0.001" placeholder="known mm">
 <select id="ax"><option value="length">length</option><option value="width">width</option></select>
</div>
<button onclick="cal()">Calibrate</button>
<button onclick="post('/log')">Log reading</button>
<button onclick="post('/profile')">Profile</button>
</div><script>
function el(i){return document.getElementById(i)}
function say(t){var v=String(t||'');el('msg').textContent=v.length>140?v.slice(0,140):v}
async function poll(){
 try{const s=await (await fetch('/state')).json();
  if(s.ok){
   const cal=s.calibrated, u=cal?'mm':'px', d=cal?2:1;
   el('L').innerHTML=(cal?s.length_mm:s.length_px).toFixed(d)+'<small>'+u+'</small>';
   el('W').innerHTML=(cal?s.width_mm:s.width_px).toFixed(d)+'<small>'+u+'</small>';
   el('T').textContent=s.tilt.toFixed(2)+'\u00b0';
   el('S').textContent=cal?s.um_per_px.toFixed(2)+' um/px':'uncal';
   el('L').classList.remove('stale'); el('W').classList.remove('stale');
  } else {
   el('L').textContent='--'; el('W').textContent='--';
   el('L').classList.add('stale'); el('W').classList.add('stale');
   say(s.err);
  }
 }catch(e){}
 setTimeout(poll,600);
}
async function post(u){say('working...');say(await (await fetch(u,{method:'POST'})).text())}
async function cal(){const v=parseFloat(el('km').value);
 if(!(v>0)){say('enter the known size first');return}
 say('calibrating...');
 const r=await fetch('/calibrate',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({known_mm:v,axis:el('ax').value})});
 say(await r.text())}
poll();
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thickness", type=float, required=True, help="specT, mm")
    p.add_argument("--raw", action="store_true",
                   help="measure the raw green plane instead of the processed frame")
    p.add_argument("--log", metavar="CSV", help="file the Log button appends to")
    p.add_argument("--note", default="", help="note stored with logged rows")
    p.add_argument("--window", type=int, default=20, help="rolling statistics depth")
    p.add_argument("--chords", type=int, default=20, help="chords per axis for Profile")
    p.add_argument("--band", type=float, default=3.0,
                   help="mm of coupon averaged into each chord; 0 = single row")
    p.add_argument("--trim", type=float, default=0.05,
                   help="fraction dropped from each end, where chords clip corners")
    p.add_argument("--profile-csv", metavar="FILE",
                   help="Profile also writes every chord to this CSV")
    p.add_argument("--exposure", type=int, help="microseconds; omit for auto")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--calib-file", default="calibration.json")
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1",
                   help="fixed ROI, bypassing auto-detection. Use this when other "
                        "features in frame - a platform edge, a bright strip - "
                        "compete with the specimen.")
    p.add_argument("--max-residual", type=float, default=0.01,
                   help="refuse above this edge-fit residual, as a fraction of the "
                        "measured span. 0.01 is strict; raise it to see a number "
                        "from a specimen whose edges are not straight, knowing it "
                        "is a straight-line fit to something that is not.")
    p.add_argument("--prominence", type=float, default=3.0,
                   help="how far the specimen's edges must stand out of the "
                        "background for auto-ROI to accept them")
    p.add_argument("--eff-dist", type=float, default=775.0,
                   help="A, front-of-lens to plate in mm. Only scales the thickness "
                        "correction, so a tape measurement is fine (see --calib-stack "
                        "in gauge.py for the exact method).")
    a = p.parse_args()

    cfg = G.Config()
    cfg.specT = a.thickness
    cfg.maxResidualFrac = a.max_residual
    cfg.effDist = a.eff_dist
    # Calibration is OPTIONAL. Without it the tool still reports pixels and the
    # quality diagnostics, which is what you need in order to set focus and
    # exposure -- and those have to be right BEFORE calibrating, or you bake a
    # blurred, clipped frame into the constant everything else depends on.
    have = G.load_calib(a.calib_file, cfg)
    if have:
        calib = json.load(open(a.calib_file))
        calib_w = int(calib.get("imageWidth", 0))
        if a.eff_dist and abs(a.eff_dist - cfg.effDistance) > 0.5:
            cfg.effDist = a.eff_dist
    else:
        calib_w = 0
        print(f"# no {a.calib_file} -- reporting PIXELS. Calibrate from the page.")
    live = {"focalPx": cfg.focalPx if have else 0.0, "calib_w": calib_w}

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (4056, 3040), "format": "RGB888"},
        # 768x576 rather than 800x600: both are multiples of 64, so the ISP adds
        # no stride padding. An unaligned width leaves padding bytes at the end of
        # every row which the YUV->BGR conversion reads as picture, and since
        # zero-filled chroma decodes to strong green, you get a green bar.
        lores={"size": (768, 576), "format": "YUV420"},
        raw={"size": (4056, 3040)} if a.raw else None,
        buffer_count=3,
    ))
    if a.exposure:
        picam2.set_controls({"AeEnable": False, "ExposureTime": a.exposure})
    picam2.start()
    time.sleep(1.5)
    rawfmt = picam2.camera_configuration()["raw"]["format"] if a.raw else None

    lock = threading.Lock()
    view = {"jpeg": None, "roi": None, "mainw": 0}
    hist = collections.deque(maxlen=a.window)
    # The measure thread writes, the HTTP threads read. Rather than share a dict
    # -- iterating one while another thread adds keys raises "dictionary changed
    # size during iteration" -- the measure thread serialises a complete snapshot
    # and publishes it as a single atomic reference swap.
    snap = {"json": json.dumps({"ok": False, "err": "starting"}).encode(),
            "cfg": None, "r": None, "ok": False}

    def publish(d, cfg_=None, r_=None):
        snap["json"] = json.dumps(d).encode()
        snap["cfg"], snap["r"], snap["ok"] = cfg_, r_, d.get("ok", False)

    def preview_loop():
        while True:
            with lock:
                yuv = picam2.capture_array("lores")
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR)
            # Overlay the ROI actually in use, scaled from measurement pixels into
            # preview pixels. Seeing the box is the difference between guessing at
            # "0 valid lines" and knowing the box is in the wrong place.
            if view["roi"] and view["mainw"]:
                k = bgr.shape[1] / view["mainw"]
                x0, y0, x1, y1 = (int(v * k) for v in view["roi"])
                cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 200, 255), 1)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                view["jpeg"] = buf.tobytes()
            time.sleep(0.06)

    def grab():
        """One frame, as float32 in the 0-255 range the pipeline expects.

        Shared by the measure loop and the profile button so both work on
        identical pixels -- a profile taken from a different capture path than
        the running measurement would be quietly comparing two things.
        """
        with lock:
            if a.raw:
                raw = picam2.capture_array("raw")
                if raw.dtype == np.uint8:
                    raw = np.ascontiguousarray(raw).view(np.uint16)
                raw = np.squeeze(raw)[:3040, :4056]
                return green_plane(raw, rawfmt) * (255.0 / (2 ** 12 - 1))
            rgb = picam2.capture_array("main")
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # The ROI may be given in PIXELS of the measured frame, or as FRACTIONS of it
    # (all four values <= 1). Fractions are the safer form: the preview you read
    # coordinates off is the small lores stream, while measurement runs on the
    # full-size main stream, so pixel coordinates taken from the preview are wrong
    # by the ratio between them -- about 5x here, which shows up as "0 valid lines".
    roi_spec = ([float(v) for v in a.roi.split(",")] if a.roi else None)
    roi_frac = roi_spec is not None and all(v <= 1.0 for v in roi_spec)

    def find_roi(img):
        if roi_spec is None:
            return G.auto_roi(img, min_prominence=a.prominence)
        h, w = img.shape
        if roi_frac:
            return (int(roi_spec[0]*w), int(roi_spec[1]*h),
                    int(roi_spec[2]*w), int(roi_spec[3]*h))
        return tuple(int(v) for v in roi_spec)

    def configure_for(img):
        c = G.replace(cfg, roi=find_roi(img))
        c.focalPx = (G.scale_focal(live["focalPx"], live["calib_w"], img.shape[1])
                     if live["focalPx"] > 0 else 1.0)
        return c

    def measure_loop():
        while True:
            t0 = time.time()
            try:
                img = grab()
                view["mainw"] = int(img.shape[1])
                c = G.replace(cfg, roi=find_roi(img))
                view["roi"] = c.roi
                calibrated = live["focalPx"] > 0
                # focalPx is in PIXELS, so it only means anything at the resolution
                # it was calibrated at. Rescale rather than silently reporting a
                # plausible-looking number that is wrong by the resolution ratio.
                c.focalPx = (G.scale_focal(live["focalPx"], live["calib_w"],
                                           img.shape[1]) if calibrated else 1.0)
                r = G.measure(img, c, a.thickness)

                hist.append((r.length_px, r.width_px))
                Lp = np.array([h[0] for h in hist]); Wp = np.array([h[1] for h in hist])
                k = (c.mmPerPx(a.thickness) if calibrated else 0.0)
                p5, p95 = np.percentile(img, 5), np.percentile(img, 95)

                publish({
                    "ok": True, "calibrated": bool(calibrated),
                    "width_px": float(r.width_px),
                    "length_px": float(r.length_px), "imgw": int(img.shape[1]),
                    "length_mm": float(r.length_px * k), "width_mm": float(r.width_px * k),
                    "Lmean": float(Lp.mean() * k), "Wmean": float(Wp.mean() * k),
                    "Lsd": float(Lp.std(ddof=1) * k) if len(Lp) > 1 else 0.0,
                    "Wsd": float(Wp.std(ddof=1) * k) if len(Wp) > 1 else 0.0,
                    "Lsd_px": float(Lp.std(ddof=1)) if len(Lp) > 1 else 0.0,
                    "n": len(hist), "frame": f"{img.shape[1]}x{img.shape[0]}",
                    "roi": "|".join(str(v) for v in c.roi),
                    "edge_rms": float(r.edge_rms_px),
                    "tilt": float(r.tilt_deg), "um_per_px": float(k * 1000),
                    "clipped": float((img >= 254).mean() * 100),
                    "p95": int(p95), "full": 255,
                    "rate": 1.0 / max(time.time() - t0, 1e-3),
                }, c, r)
            except Exception as e:
                publish({"ok": False, "err": str(e)[:90]})
                time.sleep(0.4)

    threading.Thread(target=preview_loop, daemon=True).start()
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

        def do_GET(self):
            if self.path == "/":
                self._send(HTML.encode(), "text/html")
            elif self.path == "/touch":
                self._send(TOUCH_HTML.encode(), "text/html")
            elif self.path == "/state":
                self._send(snap["json"], "application/json")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        j = view["jpeg"]
                        if j:
                            self.wfile.write(
                                b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/profile":
                try:
                    img = grab()
                    c = configure_for(img)
                    band = a.band if live["focalPx"] > 0 else 0.0
                    lp, sp = C.measure_chords(img, c, a.thickness,
                                              a.chords, band, a.trim)
                except Exception as e:
                    return self._send(f"profile failed: {e}".encode())

                cal = live["focalPx"] > 0
                out = []
                for pr in (lp, sp):
                    st = pr.summary()
                    vals = (pr.spans() if cal
                            else np.array([ch.span_px for ch in pr.chords]))
                    u = "mm" if cal else "px"
                    out.append(f"{pr.label.upper()}  {st['n']} chords, "
                               f"tilt {pr.tilt_deg:+.3f} deg")
                    out.append(f"  min {vals.min():9.4f}  max {vals.max():9.4f}  "
                               f"mean {vals.mean():9.4f} {u}")
                    rng = float(vals.max() - vals.min())
                    out.append(f"  range {rng*1000:8.1f} um"
                               if cal else f"  range {rng:8.3f} px")
                    out.append(C.sparkline(pr) if cal else "")
                    # A profile that dips at BOTH ends is usually the end chords
                    # clipping corners on a tilted coupon, not a real barrel shape.
                    if len(vals) >= 5:
                        mid = vals[len(vals)//2]
                        dip = (mid - min(vals[0], vals[-1])) / vals.mean()
                        if vals[0] < mid and vals[-1] < mid and dip > 0.002:
                            out.append(f"  both ends low by {dip*100:.2f}% "
                                       f"-- raise --trim")
                    out.append("")
                if a.profile_csv:
                    C.write_csv(a.profile_csv, (lp, sp))
                    out.append(f"wrote {a.profile_csv}")
                return self._send("\n".join(out).encode())

            if self.path == "/calibrate":
                n = int(self.headers.get("Content-Length", 0))
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                    known = float(req["known_mm"])
                    axis = req.get("axis", "length")
                except Exception as e:
                    return self._send(f"bad request: {e}".encode())
                if not snap["ok"]:
                    return self._send(b"no valid measurement to calibrate from")
                if known <= 0:
                    return self._send(b"known length must be positive")

                # Calibrate from the MEAN of the rolling window, not one frame.
                # A single frame carries the full per-frame noise straight into the
                # constant that scales every later measurement; averaging what is
                # already on screen costs nothing and divides it by sqrt(n).
                px = np.mean([h[0] if axis == "length" else h[1] for h in hist])
                s_cal = cfg.effDistance - a.thickness
                if s_cal <= 0:
                    return self._send(b"thickness exceeds eff-dist; check both")
                # float(), not numpy float: np.mean returns a numpy scalar, and
                # a numpy scalar comparison yields numpy.bool_, which json refuses.
                # It would take down the whole state endpoint at the first frame
                # after calibrating.
                f = float(px * s_cal / known)
                w = int(json.loads(snap["json"])["imgw"])

                live["focalPx"], live["calib_w"] = f, int(w)
                c2 = G.replace(cfg, focalPx=f)
                G.save_calib(a.calib_file, f, c2, image="live",
                             known_mm=known, axis=axis,
                             thickness=a.thickness, imageWidth=w,
                             frames=len(hist))
                hist.clear()
                fov = 2 * np.degrees(np.arctan((w / 2) / f))
                msg = (f"focalPx {f:.2f} from {px:.3f} px over {a.window} frames"
                       f" | {s_cal/f*1000:.4f} um/px | FOV {fov:.1f} deg"
                       f"{'' if 15 < fov < 100 else '  <-- IMPLAUSIBLE, check --eff-dist'}")
                return self._send(msg.encode())

            if self.path != "/log":
                return self.send_error(404)
            if not a.log:
                return self._send(b"no --log file given")
            if not snap["ok"]:
                return self._send(b"no valid measurement to log")
            G.log_row(a.log, snap["cfg"], snap["r"],
                      time.strftime("live_%H%M%S"), a.note)
            self._send(f"logged to {a.log}".encode())

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()

    if live["focalPx"] > 0:
        print(f"\n  focalPx {live['focalPx']:.2f}, calibrated at "
              f"{live['calib_w'] or '?'} px wide")
    else:
        print("\n  uncalibrated -- reporting pixels until you calibrate from the page")
    print(f"  desktop     http://{ip}:{a.port}")
    print(f"  touchscreen http://{ip}:{a.port}/touch")
    print(f"  (Ctrl-C to stop)\n")
    try:
        http.server.ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop(); picam2.close()


if __name__ == "__main__":
    main()