#!/usr/bin/env python3
"""
pi_dual.py -- two HQ cameras: bottom for length and width, side for thickness.

    python3 pi_dual.py --eff-dist0 775 --eff-dist1 300
    python3 pi_dual.py --eff-dist0 775 --eff-dist1 300 --log runs.csv
"""

import argparse
import csv
import http.server
import json
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2

import chord_measure as C
import gauge as G

try:                      # optional: push each logged row to the PaperSteel app
    from identifier_push import send_measurement
except Exception:         # not installed, or no stark_ingest.conf yet
    send_measurement = None
try:
    import specimen_registry as SR
except Exception:
    SR = None


# ------------------------------------------------------------------ camera --

class Cam:
    """The HARDWARE only: one sensor, its streams, its preview buffer."""

    def __init__(self, index, exposure, size=(4056, 3040)):
        self.index = index
        self.lock = threading.Lock()
        self.jpeg = None
        self.pc = Picamera2(index)
        self.pc.configure(self.pc.create_video_configuration(
            main={"size": size, "format": "RGB888"},
            # 64-aligned, so the ISP adds no stride padding for the preview to misread
            # as picture.
            lores={"size": (512, 384), "format": "YUV420"},
            buffer_count=2,
        ))
        if exposure:
            self.pc.set_controls({"AeEnable": False, "ExposureTime": exposure})
        self.pc.start()

    def set_exposure(self, us):
        """Fix the exposure, or hand it back to the sensor's own metering.

        Blank or 0 re-enables AE. Auto exposure is convenient for aiming and
        poison for measuring: the ISP walks the backlight level between frames,
        which moves both edges of the silhouette.
        """
        with self.lock:
            if us:
                self.pc.set_controls({"AeEnable": False,
                                      "ExposureTime": int(us)})
            else:
                self.pc.set_controls({"AeEnable": True})

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


class Window:
    """A rolling buffer whose length can be changed while the server runs.

    collections.deque fixes maxlen at construction, and the handlers close over
    this object, so a deque could never be resized from the settings panel -- a
    nested function cannot rebind a name in main(). This keeps the same tiny
    interface (append / clear / len / np.array) and adds resize().
    """

    def __init__(self, n):
        self.v = []
        self.resize(n)

    def resize(self, n):
        self.n = max(1, int(n))
        del self.v[:-self.n]

    def append(self, x):
        self.v.append(x)
        del self.v[:-self.n]

    def clear(self):
        self.v.clear()

    def __len__(self):
        return len(self.v)

    def __array__(self, dtype=None):
        return np.asarray(self.v, dtype)


class Role:
    """A measurement JOB -- bottom or side -- with its own standoff, ROI and
    calibration, pointed at whichever Cam currently performs it."""

    def __init__(self, name, eff_dist, roi, prominence, calib_file, max_resid,
                 ref_file=None, datum=None):
        self.name, self.calib_file = name, calib_file
        self.datum = None if datum in (None, "none") else datum
        self.datum_off = 0.0     # px between the reference datum and true contact
        self.extras = None
        self.ref_file = ref_file
        self.ref = None          # G.Reference of the EMPTY fixture, or None
        self.touch = 0           # specimen pixels pressed against opaque fixture
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
                rec = json.load(open(calib_file))
                self.calib_w = int(rec.get("imageWidth", 0))
                self.datum_off = float(rec.get("datumOffsetPx", 0.0))
            except Exception:
                pass
            if eff_dist:
                self.cfg.effDist = eff_dist
        else:
            print(f"# {name}: no {calib_file} -- reporting pixels")
        self.roi_spec = ([float(v) for v in roi.split(",")] if roi else None)
        self.roi_frac = (self.roi_spec is not None
                         and all(v <= 1.0 for v in self.roi_spec))
        if ref_file and os.path.exists(ref_file):
            try:
                self.ref = G.Reference(np.load(ref_file))
                print(f"# {name}: reference loaded from {ref_file} "
                      f"({self.ref.masked_frac*100:.1f}% of frame opaque)")
            except Exception as e:
                print(f"# {name}: could not load {ref_file}: {e}")

    def grab(self):
        """A frame with the empty fixture divided out, when a reference exists."""
        img = self.cam.grab()
        if self.ref is None:
            self.touch = 0
            return img
        # Count contact only inside the ROI -- the only region that can change
        # the result. Before the first successful measurement there is no ROI
        # yet, so record the figure but do not refuse on it.
        img, self.touch, ring_n = self.ref.apply(img, self.last_roi)
        # A RATIO, not a count. The ring follows every hole in the mask, and in
        # the fixture region the division gain is large enough that noise alone
        # reads dark -- tens of thousands of pixels nowhere near the specimen.
        # An edge genuinely buried in the fixture darkens most of the ring it
        # touches, so a quarter of it is a wide margin either way.
        frac = (self.touch / ring_n) if ring_n else 0.0
        if (self.last_roi and not self.datum
                and ring_n > 200 and frac > 0.25):
            raise RuntimeError(
                f"{self.name}: coupon edge is behind the opaque part of the "
                f"fixture ({frac*100:.0f}% of the boundary, {self.touch} of "
                f"{ring_n} px) - reseat it so the edge is visible")
        return img

    def capture_reference(self, n):
        """Average n frames of the EMPTY fixture.

        Accumulated in place. Holding all n frames and handing them to np.mean
        costs 2 * n * 49 MB at 4056x3040 float32 -- 1.6 GB at the default n=16,
        which the OOM killer ends on a Pi also running two cameras and Chromium.
        This way the peak is two frames, whatever n is. Summing 16 frames of
        0..255 reaches 4080, so float32 carries it exactly.
        """
        acc = None
        hottest = 0.0
        for _ in range(n):
            f = self.cam.grab()
            hottest = max(hottest, float(f.max()))
            if acc is None:
                acc = f                  # adopt the first frame's buffer
            else:
                acc += f                 # in place: allocates nothing
                del f
        acc /= float(n)
        clipped = float((acc >= 254).mean() * 100)
        if hottest >= 255.0 and clipped < 0.01:
            # Averaging hides clipping that was present in individual frames.
            clipped = max(clipped, 0.02)
        self.ref = G.Reference(acc)
        if self.ref_file:
            np.save(self.ref_file, acc)
        return self.ref, clipped

    def clear_reference(self):
        self.ref = None
        if self.ref_file and os.path.exists(self.ref_file):
            os.remove(self.ref_file)

    def set_roi(self, spec):
        """Change the ROI while running. spec is 'x0,y0,x1,y1' -- pixels, or
        fractions of the frame when every value is <= 1 -- or blank for auto."""
        if spec is None or not str(spec).strip() or str(spec).strip().lower() in (
                "auto", "none"):
            self.roi_spec, self.roi_frac = None, False
            return "auto"
        vals = [float(v) for v in str(spec).replace(" ", "").split(",")]
        if len(vals) != 4:
            raise ValueError("need four numbers: x0,y0,x1,y1")
        if vals[0] >= vals[2] or vals[1] >= vals[3]:
            raise ValueError("need x0 < x1 and y0 < y1")
        if min(vals) < 0:
            raise ValueError("values cannot be negative")
        self.roi_spec = vals
        self.roi_frac = all(v <= 1.0 for v in vals)
        return "fractions" if self.roi_frac else "pixels"

    def find_roi(self, img):
        """Resolve the ROI and remember it for the preview overlay."""
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
        """The Config chord_measure.py needs: this role's ROI and calibration, with focalPx
        rescaled to whatever resolution the frame actually is."""
        c = G.replace(self.cfg, roi=self.find_roi(img))
        c.focalPx = (G.scale_focal(self.focalPx, self.calib_w, img.shape[1])
                     if self.focalPx > 0 else 1.0)
        c.specT = thickness
        return c

    def measure(self, img, thickness):
        """`thickness` is the offset of this role's silhouette plane from its reference:
        specT for the bottom view, 0 for the side view, whose reference face IS its
        silhouette plane."""
        c = G.replace(self.cfg, roi=self.find_roi(img))
        c.focalPx = (G.scale_focal(self.focalPx, self.calib_w, img.shape[1])
                     if self.focalPx > 0 else 1.0)
        if self.datum:
            if self.ref is None:
                raise RuntimeError(f"{self.name}: datum mode needs an empty-fixture "
                                   f"reference - capture one")
            r, self.extras = G.measure_datum(img, self.ref.ref, c, thickness,
                                             self.datum, self.datum_off)
        else:
            r = G.measure(img, c, thickness)
        k = c.mmPerPx(thickness) if self.focalPx > 0 else 0.0
        return r, c, k

    def skew(self):
        """Degrees between the fixture's datum line and the specimen's edge.

        Zero when the specimen is square to the fixture. It is reported because
        a datum measurement silently degrades with skew and nothing else in the
        output shows it: the edge fits stay perfectly straight either way.
        """
        if not self.datum or not self.extras:
            return None
        v = self.extras.get("skew_deg")
        return None if v is None else float(v)

    def calibrate(self, px, known_mm, axis, thickness, width):
        if not np.isfinite(px):
            raise ValueError(f"{self.name}: {axis} not measurable in this frame")
        s_cal = self.cfg.effDistance - thickness
        if s_cal <= 0:
            raise ValueError("thickness exceeds eff-dist")
        if self.datum and axis == "width":
            # On a datum view the small axis sets the zero offset, not the scale.
            if self.focalPx <= 0:
                raise ValueError(f"{self.name}: calibrate the LENGTH first - the "
                                 f"datum offset is set against the scale")
            mmpx = s_cal / G.scale_focal(self.focalPx, self.calib_w, width)
            self.datum_off = float(known_mm / mmpx - (px - self.datum_off))
            G.save_calib(self.calib_file, self.focalPx,
                         G.replace(self.cfg, focalPx=self.focalPx),
                         image=f"live-{self.name}", imageWidth=int(self.calib_w),
                         datumOffsetPx=self.datum_off, datumKnownMm=known_mm)
            return (f"{self.name}: datum offset {self.datum_off:+.3f} px "
                    f"({self.datum_off*mmpx*1000:+.1f} um)")
        f = float(px * s_cal / known_mm)
        self.focalPx, self.calib_w = f, int(width)
        G.save_calib(self.calib_file, f, G.replace(self.cfg, focalPx=f),
                     image=f"live-{self.name}", known_mm=known_mm, axis=axis,
                     thickness=thickness, imageWidth=int(width),
                     datumOffsetPx=self.datum_off)
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
#msg{color:#9c9;font-size:12px;min-height:34px;margin-top:8px;white-space:pre-wrap;word-break:break-word}
details{margin-top:14px;border:1px solid #333;border-radius:6px;padding:4px 10px 10px}
summary{cursor:pointer;font-size:12px;text-transform:uppercase;letter-spacing:.07em;
color:#7d7d7d;padding:8px 0}
.cfgrow{display:flex;align-items:flex-start;gap:10px;margin:7px 0;padding-bottom:7px;
border-bottom:1px solid #2a2a2a}
.cfgrow label{flex:1;font-size:13px;color:#ccc}
.cfgrow input,.cfgrow select{width:170px;flex:none}
.combo{position:relative;display:inline-block}
.droplist{display:none;position:absolute;z-index:9;left:0;top:calc(100% + 2px);
min-width:200px;max-height:280px;overflow:auto;background:#242424;
border:1px solid #3a3a3a;border-radius:6px;box-shadow:0 6px 18px #0009}
.dropitem{padding:7px 11px;cursor:pointer;font:13px ui-monospace,monospace;
white-space:nowrap}
.dropitem:hover{background:#2d7d5a}
.dropitem.dim{color:#888;cursor:default}
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
  <div class="lab">Specimen</div>
  <select id="smode" onchange="smode()">
   <option value="existing">existing</option><option value="new">new</option></select>
  <span class="combo"><input id="sid" placeholder="PS-V4-038" style="width:170px"
    autocomplete="off" oninput="this.value=this.value.toUpperCase();filtersp()"
    onfocus="opensp()" onblur="setTimeout(hidesp,180)"
   ><div id="splist" class="droplist"></div></span>
  <button onclick="getspecimens()" style="background:#555">Refresh list</button>
  <button onclick="logrow()">Log</button>
  <div class="lab">Region of interest</div>
  <div class="sub">x0,y0,x1,y1 -- pixels, or fractions of the frame when all are
   &le; 1. Blank for auto. Uses the view selected above.</div>
  <input id="roi" placeholder="0.17,0.10,0.97,0.44" style="width:240px">
  <button onclick="setroi()">Set ROI</button>
  <button onclick="el('roi').value='';setroi()" style="background:#555">Auto</button>
  <div class="lab">Empty-fixture reference</div>
  <div class="sub">Remove the coupon, then capture. Divides out the fixture and the
   backlight's unevenness. Uses the view selected above.</div>
  <button onclick="ref('capture')">Capture reference</button>
  <button onclick="ref('clear')" style="background:#555">Clear reference</button>
  <button onclick="post('/profile')">Profile</button>
  <button onclick="if(confirm('Swap the cameras? Both calibrations will need redoing.'))post('/swap')"
   style="background:#7d5a2d">Swap cameras</button>
  <details id="cfg" ontoggle="if(this.open)loadcfg()">
   <summary>Settings</summary>
   <div class="sub">Everything the command line can set. A flag you passed at
    launch stays in force until you change it here; what you change here is
    saved to the state file and survives a restart.</div>
   <div id="cfgbody"><div class="sub">loading...</div></div>
   <button onclick="savecfg()">Apply &amp; save</button>
   <button onclick="loadcfg()" style="background:#555">Revert</button>
  </details>
  <div id="msg"></div>
  <pre id="pf" style="background:#111;padding:9px;border-radius:5px;font-size:11px;
   white-space:pre;overflow-x:auto;color:#bbb;margin-top:8px"></pre>
 </div>
</div><script>
function el(i){return document.getElementById(i)}
function say(t){var v=String(t||'');el('msg').textContent=v.length>600?v.slice(0,600):v}
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
   (s.skew0===null||s.skew0===undefined?'':
    row('bottom datum skew',s.skew0.toFixed(3)+' deg',
        Math.abs(s.skew0)>0.05?'warn':'ok'))+
   (s.skew1===null||s.skew1===undefined?'':
    row('side datum skew',s.skew1.toFixed(3)+' deg',
        Math.abs(s.skew1)>0.05?'warn':'ok'))+
   row('side tilt',s.tilt1===null?'--':s.tilt1.toFixed(2)+' deg')+
   row('pairing','bottom=sensor '+s.sensor0+', side=sensor '+s.sensor1,
       s.swapped?'warn':'')+
   row('bottom ROI',s.roi0?s.roi0.join(', '):'--')+
   row('side ROI',s.roi1?s.roi1.join(', '):'--')+
   row('bottom reference',s.ref0?'on':'off',s.ref0?'ok':'warn')+
   row('bottom datum',s.datum0?s.datum0+', offset '+s.doff0.toFixed(2)+' px':'none (two edges)')+
   row('side datum',s.datum1?s.datum1+', offset '+s.doff1.toFixed(2)+' px':'none (two edges)')+
   row('side reference',s.ref1?'on':'off',s.ref1?'ok':'warn')+
   (s.touch0>0&&!s.datum0?
    row('bottom','coupon touching opaque fixture - edge hidden','bad'):'')+
   (s.touch1>0&&!s.datum1?
    row('side','coupon touching opaque fixture - edge hidden','bad'):'')+
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
function cfgesc(v){return String(v===null||v===undefined?'':v)
 .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;')}
var CFGHEAD={live:'takes effect immediately',camera:'sent to the sensor now',
 restart:'needs a restart'};
async function loadcfg(){
 try{const r=await (await fetch('/settings')).json();var g={},o=[];
  r.spec.forEach(function(t){(g[t.scope]=g[t.scope]||[]).push(t)});
  ['live','camera','restart'].forEach(function(sc){
   if(!g[sc])return;
   o.push('<div class="lab">'+CFGHEAD[sc]+'</div>');
   g[sc].forEach(function(t){
    var v=r.values[t.key],p=r.pending[t.key],shown=(p===undefined?v:p);
    var f=t.choices?('<select id="cf_'+t.key+'">'+t.choices.map(function(c){
      return '<option'+(c===String(v)?' selected':'')+'>'+c+'</option>'
     }).join('')+'</select>')
     :('<input id="cf_'+t.key+'" value="'+cfgesc(shown)+'">');
    o.push('<div class="cfgrow"><label for="cf_'+t.key+'">'+t.label+
     (p===undefined?'':' <b style="color:#e8a33d">pending restart</b>')+
     (t.hint?'<br><span class="sub">'+t.hint+'</span>':'')+'</label>'+f+'</div>')})});
  el('cfgbody').innerHTML=o.join('')}
 catch(e){el('cfgbody').innerHTML='<div class="sub">could not load settings</div>'}}
async function savecfg(){var o={};
 document.querySelectorAll('[id^=cf_]').forEach(function(e){o[e.id.slice(3)]=e.value});
 say('saving...');
 say(await (await fetch('/settings',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(o)})).text());
 loadcfg()}
var SPEC=[];
function smode(){var n=el('smode').value==='new';
 el('sid').placeholder=n?'PS-V4   (series)':'PS-V4-038';
 hidesp()}
function hidesp(){el('splist').style.display='none'}
function picksp(c){el('sid').value=c;hidesp()}
function opensp(){if(!SPEC.length){getspecimens(1)}else{filtersp()}}
function filtersp(){
 if(el('smode').value!=='existing'){return hidesp()}
 var q=el('sid').value.trim().toUpperCase(),o;
 var m=SPEC.filter(function(c){return c.indexOf(q)>=0});
 if(m.length){o=m.slice(0,60).map(function(c){
   return '<div class="dropitem" onmousedown="picksp(\''+c+'\')">'+c+'</div>'}).join('')
   +(m.length>60?'<div class="dropitem dim">'+(m.length-60)+' more - keep typing</div>':'')}
 else{o='<div class="dropitem dim">'+(SPEC.length?'no match':'list not loaded')+'</div>'}
 el('splist').innerHTML=o;el('splist').style.display='block'}
async function getspecimens(quiet){if(!quiet){say('fetching specimen list...')}
 try{const r=await (await fetch('/specimens')).json();
  SPEC=r.codes||[];
  if(r.error){say('list failed: '+r.error)}
  else if(!quiet){say(SPEC.length+' specimens loaded')}
  filtersp()}
 catch(e){say('could not reach the app')}}
async function logrow(){
 say('logging...');
 say(await (await fetch('/log',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:el('smode').value,
   specimen_id:el('sid')?el('sid').value.trim():''})})).text())}
// Side view: label the small axis 'thickness' (value stays 'width').
function axes(){var side=el('cam').value==='1';
 el('ax').options[1].text=side?'thickness':'width'}
async function cal(){const v=parseFloat(el('km').value);
 if(!(v>0)){say('enter the known size first');return}
 say('calibrating...');
 say(await (await fetch('/calibrate',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({known_mm:v,camera:parseInt(el('cam').value),
   axis:el('ax').value})})).text())}
async function setroi(){say('setting ROI...');
 say(await (await fetch('/roi',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({camera:parseInt(el('cam').value),
   roi:el('roi').value})})).text())}
async function ref(action){
 if(action==='capture'&&!confirm('Is the fixture EMPTY? The reference must not contain a coupon.'))return;
 say(action==='capture'?'capturing reference...':'clearing...');
 say(await (await fetch('/reference',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({camera:parseInt(el('cam').value),action:action})})).text())}
axes();poll();
</script></body></html>"""

TOUCH = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>gauge</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>*{box-sizing:border-box;-webkit-user-select:none;user-select:none;
touch-action:manipulation}
html,body{margin:0;height:100%;background:#101010;color:#eee;font:16px system-ui;
overflow:hidden}
.pane{height:100%;display:flex;flex-direction:column;padding:16px;gap:6px;
overflow-y:auto}
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
#msg{min-height:38px;font-size:15px;color:#8fbf9f;overflow:auto;max-height:22vh;white-space:pre-wrap;word-break:break-word}
.sub{font-size:13px;color:#777}
details{margin-top:9px;border:1px solid #333;border-radius:11px;padding:4px 12px 10px;
max-height:46vh;overflow:auto}
summary{cursor:pointer;font-size:17px;letter-spacing:.13em;color:#7d7d7d;
text-transform:uppercase;padding:10px 0}
.cfgrow{display:flex;align-items:center;gap:10px;margin:8px 0}
.cfgrow label{flex:1;font-size:15px;color:#ccc}
.cfgrow input,.cfgrow select{width:150px;flex:none;min-height:50px;font-size:20px}
.combo{position:relative}
.droplist{display:none;position:absolute;z-index:9;left:0;bottom:calc(100% + 4px);
width:100%;max-height:44vh;overflow:auto;background:#242424;
border:1px solid #3a3a3a;border-radius:11px;box-shadow:0 6px 22px #000b}
.dropitem{padding:15px 14px;font:21px ui-monospace,monospace;
border-bottom:1px solid #2e2e2e}
.dropitem.dim{color:#888}
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
<div class="row" style="margin-top:9px">
 <select id="smode" onchange="smode()" style="flex:0 0 38%">
  <option value="existing">existing</option><option value="new">new</option></select>
 <span class="combo" style="flex:1"><input id="sid" placeholder="PS-V4-038"
   style="width:100%" autocomplete="off"
   oninput="this.value=this.value.toUpperCase();filtersp()"
   onfocus="opensp()" onblur="setTimeout(hidesp,250)"
  ><div id="splist" class="droplist"></div></span>
</div>
<button onclick="logrow()">Log reading</button>
<button onclick="getspecimens()" style="background:#555">Refresh specimen list</button>
<button onclick="post('/profile')">Profile</button>
<button onclick="if(confirm('Swap cameras? Recalibrate after.'))post('/swap')"
 style="background:#7d5a2d">Swap cameras</button>
<details id="cfg" ontoggle="if(this.open)loadcfg()">
 <summary>Settings</summary>
 <div class="sub">Saved to the state file; survives a restart.</div>
 <div id="cfgbody"><div class="sub">loading...</div></div>
 <button onclick="savecfg()">Apply &amp; save</button>
 <button onclick="loadcfg()" style="background:#555">Revert</button>
</details>
</div><script>
function el(i){return document.getElementById(i)}
function say(t){var v=String(t||'');el('msg').textContent=v.length>400?v.slice(0,400):v}
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
function cfgesc(v){return String(v===null||v===undefined?'':v)
 .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;')}
var CFGHEAD={live:'takes effect immediately',camera:'sent to the sensor now',
 restart:'needs a restart'};
async function loadcfg(){
 try{const r=await (await fetch('/settings')).json();var g={},o=[];
  r.spec.forEach(function(t){(g[t.scope]=g[t.scope]||[]).push(t)});
  ['live','camera','restart'].forEach(function(sc){
   if(!g[sc])return;
   o.push('<div class="lab">'+CFGHEAD[sc]+'</div>');
   g[sc].forEach(function(t){
    var v=r.values[t.key],p=r.pending[t.key],shown=(p===undefined?v:p);
    var f=t.choices?('<select id="cf_'+t.key+'">'+t.choices.map(function(c){
      return '<option'+(c===String(v)?' selected':'')+'>'+c+'</option>'
     }).join('')+'</select>')
     :('<input id="cf_'+t.key+'" value="'+cfgesc(shown)+'">');
    o.push('<div class="cfgrow"><label for="cf_'+t.key+'">'+t.label+
     (p===undefined?'':' <b style="color:#e8a33d">pending restart</b>')+
     (t.hint?'<br><span class="sub">'+t.hint+'</span>':'')+'</label>'+f+'</div>')})});
  el('cfgbody').innerHTML=o.join('')}
 catch(e){el('cfgbody').innerHTML='<div class="sub">could not load settings</div>'}}
async function savecfg(){var o={};
 document.querySelectorAll('[id^=cf_]').forEach(function(e){o[e.id.slice(3)]=e.value});
 say('saving...');
 say(await (await fetch('/settings',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(o)})).text());
 loadcfg()}
var SPEC=[];
function smode(){var n=el('smode').value==='new';
 el('sid').placeholder=n?'PS-V4   (series)':'PS-V4-038';
 hidesp()}
function hidesp(){el('splist').style.display='none'}
function picksp(c){el('sid').value=c;hidesp()}
function opensp(){if(!SPEC.length){getspecimens(1)}else{filtersp()}}
function filtersp(){
 if(el('smode').value!=='existing'){return hidesp()}
 var q=el('sid').value.trim().toUpperCase(),o;
 var m=SPEC.filter(function(c){return c.indexOf(q)>=0});
 if(m.length){o=m.slice(0,60).map(function(c){
   return '<div class="dropitem" onmousedown="picksp(\''+c+'\')">'+c+'</div>'}).join('')
   +(m.length>60?'<div class="dropitem dim">'+(m.length-60)+' more - keep typing</div>':'')}
 else{o='<div class="dropitem dim">'+(SPEC.length?'no match':'list not loaded')+'</div>'}
 el('splist').innerHTML=o;el('splist').style.display='block'}
async function getspecimens(quiet){if(!quiet){say('fetching specimen list...')}
 try{const r=await (await fetch('/specimens')).json();
  SPEC=r.codes||[];
  if(r.error){say('list failed: '+r.error)}
  else if(!quiet){say(SPEC.length+' specimens loaded')}
  filtersp()}
 catch(e){say('could not reach the app')}}
async function logrow(){
 say('logging...');
 say(await (await fetch('/log',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:el('smode').value,
   specimen_id:el('sid')?el('sid').value.trim():''})})).text())}
// Side view: label the small axis 'thickness' (value stays 'width').
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


LOG_COLS = ["when", "specimen_id", "length_mm", "width_mm", "thickness_mm", "length_side_mm",
            "delta_mm", "length_px", "width_px", "thickness_px", "length_side_px",
            "skew0_deg", "skew1_deg", "tilt0_deg", "tilt1_deg", "rms0_px", "rms1_px",
            "focalPx0", "effDist0", "focalPx1", "effDist1", "roi0", "roi1", "note"]


def parse_fields(spec):
    """'species=beech, batch=B12, density=720' -> a dict for the insert.

    Values go through json.loads first, so 720 arrives as a number and true as
    a boolean; anything that is not valid JSON stays the text you typed. That
    matters because a numeric column refuses the string "720" on some setups
    and accepts the number on all of them.
    """
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{part!r} is not key=value")
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"{part!r} has no column name")
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


HERE = os.path.dirname(os.path.abspath(__file__))


def here(name):
    """A default path NEXT TO THIS SCRIPT, not next to wherever you ran it.

    Launched from the wrong directory, a bare 'measurements.csv' quietly starts
    a second log somewhere else and the first one stops growing. An explicit
    --flag is still taken exactly as typed.
    """
    return os.path.join(HERE, name)


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
    p.add_argument("--log", metavar="CSV", default=here("measurements.csv"),
                   help="every logged reading is appended here. Defaults to "
                        "measurements.csv beside this script; pass --log '' to "
                        "turn the CSV off")
    p.add_argument("--note", default="")
    p.add_argument("--new-fields", default="",
                   help="extra columns written into `specimens` when the Pi "
                        "mints a new ID, as key=value pairs: "
                        "--new-fields 'species=beech, batch=B12'. Needed for "
                        "every NOT NULL column the table has besides the code.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--calib0", default=here("calib_bottom.json"))
    p.add_argument("--calib1", default=here("calib_side.json"))
    # Both views see the same corner bar, so both default the same way. In the
    # chair the coupon rests in a 90 degree corner: the bottom camera's width
    # axis ends on the vertical plate, the side camera's thickness axis ends on
    # the horizontal plate, and in both frames that bar lies BELOW the coupon.
    # Pass none for a view whose small axis has two free edges.
    p.add_argument("--datum0", default="below", choices=("none",) + G.DATUMS,
                   help="bottom view: side of the image where the coupon rests on "
                        "the fixture, or none to measure both edges (default below)")
    p.add_argument("--datum1", default="below", choices=("none",) + G.DATUMS,
                   help="side view: side of the image where the coupon rests on "
                        "the plate, or none to measure both edges (default below)")
    p.add_argument("--ref0", default=here("ref_bottom.npy"),
                   help="empty-fixture reference for the bottom view")
    p.add_argument("--ref1", default=here("ref_side.npy"),
                   help="empty-fixture reference for the side view")
    p.add_argument("--ref-frames", type=int, default=16,
                   help="frames averaged into a reference; its noise lands on "
                        "every later measurement, so more is better")
    p.add_argument("--state", default=here("dual_state.json"),
                   help="remembers which sensor is bottom and which is side")
    p.add_argument("--swap", action="store_true",
                   help="start with the two cameras exchanged")
    a = p.parse_args()

    # Persist the pairing, the ROIs and the settings so a restart can't
    # silently revert them.
    def load_state():
        try:
            return json.load(open(a.state))
        except Exception:
            return {}

    state = load_state()

    # A flag you actually typed always beats a stored value. Without this, a
    # number dialled into the panel three sessions ago would quietly override a
    # deliberate command line, which is the worst kind of bug to chase: the
    # launch line says one thing and the machine does another.
    typed = {arg.split("=", 1)[0].lstrip("-").replace("-", "_")
             for arg in sys.argv[1:] if arg.startswith("--")}
    tuned = dict(state.get("settings") or {})
    dropped = []
    for k in list(tuned):
        if k not in vars(a):
            continue
        if k in typed:
            # Typing the flag is the more deliberate act, so it wins -- and the
            # stored value is FORGOTTEN rather than left to reappear at the next
            # launch. A state file that quietly disagrees with the command line
            # is the worst kind of thing to debug.
            dropped.append(f"{k} (was {tuned.pop(k)}, now {getattr(a, k)})")
        else:
            setattr(a, k, tuned[k])

    cams = [Cam(0, a.exposure0), Cam(1, a.exposure1)]
    bottom = Role("bottom", a.eff_dist0, a.roi0, a.prominence,
                  a.calib0, a.max_residual, a.ref0, a.datum0)
    side = Role("side", a.eff_dist1, a.roi1, a.prominence,
                a.calib1, a.max_residual, a.ref1, a.datum1)

    swapped = {"v": a.swap or bool(state.get("swapped", False))}

    def bind():
        i, j = (1, 0) if swapped["v"] else (0, 1)
        bottom.cam, side.cam = cams[i], cams[j]

    def save_state():
        try:
            json.dump({"swapped": swapped["v"],
                       "roi0": bottom.roi_spec,
                       "roi1": side.roi_spec,
                       "settings": tuned}, open(a.state, "w"), indent=1)
        except Exception as e:
            print(f"# could not save {a.state}: {e}")

    # A --roi flag on the command line beats the stored one; otherwise the ROI
    # you dialled in last session is still there.
    for role, cli, key in ((bottom, a.roi0, "roi0"), (side, a.roi1, "roi1")):
        if cli is None and state.get(key):
            try:
                role.set_roi(",".join(str(v) for v in state[key]))
                print(f"# {role.name}: ROI {role.roi_spec} restored "
                      f"from {a.state}")
            except Exception as e:
                print(f"# {role.name}: ignoring bad saved ROI: {e}")

    if dropped:
        print("# command line overrides stored settings, forgetting: "
              + ", ".join(dropped))
        save_state()

    bind()
    time.sleep(1.5)

    hist = Window(a.window)

    # ------------------------------------------------------------ settings --
    # Every flag the browser can reach. "scope" says when a change bites:
    #   live     -- the next measurement already uses it
    #   camera   -- pushed to the sensor immediately
    #   restart  -- stored now, picked up when the process next starts
    # Nothing here is a file the station writes by itself; the paths are listed
    # so you can see what is in use, and changing one is a deliberate act.
    def T(key, label, kind, scope, hint, choices=None, rng=None, blank=None):
        return {"key": key, "label": label, "kind": kind, "scope": scope,
                "hint": hint, "choices": choices, "rng": rng, "blank": blank}

    DATUMS = ("none",) + tuple(G.DATUMS)
    TUNABLES = [
        T("eff_dist0", "bottom lens distance", "float", "live",
          "platform to lens, mm", rng=(1, 100000)),
        T("eff_dist1", "side lens distance", "float", "live",
          "coupon reference face to lens, mm", rng=(1, 100000)),
        T("plane_offset", "fixed silhouette plane", "float", "live",
          "mm above the bottom view's reference. 0 for the chair geometry. "
          "Blank = float, taking the thickness the side view measures",
          blank=True),
        T("datum0", "bottom datum", "choice", "live",
          "side of the frame the coupon rests against, or none for two edges",
          choices=DATUMS),
        T("datum1", "side datum", "choice", "live",
          "same, for the side view", choices=DATUMS),
        T("exposure0", "sensor 0 exposure", "int", "camera",
          "microseconds. Blank = auto, which drifts between frames and moves "
          "both edges -- fix it before measuring", blank=True, rng=(10, 2000000)),
        T("exposure1", "sensor 1 exposure", "int", "camera",
          "microseconds. Blank = auto", blank=True, rng=(10, 2000000)),
        T("prominence", "ROI prominence", "float", "live",
          "how far an edge must stand out of the background for auto-ROI to "
          "accept it", rng=(0.1, 1000)),
        T("max_residual", "max edge residual", "float", "live",
          "fraction of the span an edge fit may deviate before the view is "
          "called bad", rng=(0.0001, 1)),
        T("window", "averaging window", "int", "live",
          "readings behind the 1s thickness stability figure", rng=(1, 5000)),
        T("chords", "profile chords", "int", "live",
          "chords per axis when you press Profile", rng=(2, 2000)),
        T("band", "chord band", "float", "live",
          "mm of specimen averaged into each chord; 0 = a single row",
          rng=(0, 1000)),
        T("trim", "chord end trim", "float", "live",
          "fraction dropped at each end, where chords clip the corners",
          rng=(0, 0.45)),
        T("order", "centreline order", "int", "live",
          "polynomial order; 1 assumes perfectly straight edges", rng=(1, 8)),
        T("ref_frames", "reference frames", "int", "live",
          "frames averaged into an empty-fixture reference. Its noise lands on "
          "every later measurement, so more is better", rng=(1, 200)),
        T("note", "note on every row", "text", "live",
          "free text written into the CSV and pushed with each measurement"),
        T("new_fields", "new specimen fields", "text", "live",
          "key=value pairs written into the specimens table when you mint a "
          "new ID, e.g. species=beech, batch=B12. Every NOT NULL column that "
          "table has besides the code must appear here", blank=True),
        T("log", "measurement CSV", "text", "live",
          "appended to on every Log. Blank turns the CSV off", blank=True),
        T("profile_csv", "profile CSV", "text", "live",
          "Profile also writes every chord here. Blank = off", blank=True),
        T("calib0", "bottom calibration file", "text", "restart", "", blank=True),
        T("calib1", "side calibration file", "text", "restart", "", blank=True),
        T("ref0", "bottom reference file", "text", "restart", "", blank=True),
        T("ref1", "side reference file", "text", "restart", "", blank=True),
        T("state", "state file", "text", "restart",
          "pairing, ROIs and these settings", blank=True),
        T("port", "web port", "int", "restart", "", rng=(1, 65535)),
    ]
    BY_KEY = {t["key"]: t for t in TUNABLES}

    def coerce(t, raw):
        """Text from a form field -> the value the flag expects, or ValueError."""
        txt = str(raw).strip()
        if t["kind"] == "choice":
            if txt not in t["choices"]:
                raise ValueError(f"must be one of {', '.join(t['choices'])}")
            return txt
        if t["kind"] == "text":
            if txt == "" and not t["blank"]:
                raise ValueError("cannot be empty")
            return txt
        if txt == "":
            if not t["blank"]:
                raise ValueError("needs a number")
            return None
        v = float(txt)
        if t["kind"] == "int":
            v = int(round(v))
        lo, hi = t["rng"] or (None, None)
        if lo is not None and not (lo <= v <= hi):
            raise ValueError(f"must be between {lo} and {hi}")
        return v

    def apply_live(key, val):
        """Push a setting into the objects already running.

        Anything read straight off `a` at use time needs nothing more than the
        setattr; the rest is copied into the object that cached it at startup.
        """
        setattr(a, key, val)
        if key == "eff_dist0":
            bottom.cfg.effDist = val
        elif key == "eff_dist1":
            side.cfg.effDist = val
        elif key == "max_residual":
            bottom.cfg.maxResidualFrac = side.cfg.maxResidualFrac = val
        elif key == "prominence":
            bottom.prominence = side.prominence = val
        elif key == "datum0":
            bottom.datum = None if val == "none" else val
        elif key == "datum1":
            side.datum = None if val == "none" else val
        elif key == "window":
            hist.resize(val)
        elif key == "exposure0":
            cams[0].set_exposure(val)
        elif key == "exposure1":
            cams[1].set_exposure(val)

    def settings_doc():
        # "values" is what the station is RUNNING on. "pending" is what a
        # restart-scope field has been set to but cannot use yet -- shown
        # separately so a change never looks like it was swallowed.
        vals = {t["key"]: getattr(a, t["key"]) for t in TUNABLES}
        pend = {k: v for k, v in tuned.items()
                if k in BY_KEY and BY_KEY[k]["scope"] == "restart"
                and v != vals.get(k)}
        return {"values": vals, "pending": pend,
                "spec": [{k: t[k] for k in
                          ("key", "label", "kind", "scope", "hint", "choices")}
                         for t in TUNABLES]}
    # The most recent thickness cam1 produced, independent of whether cam0 is
    # calibrated.
    latest = {"thickness": None}
    snap = {"json": json.dumps({"rate": 0.0}).encode(), "row": None}

    def _row(r0, r1, c0, c1, specT, side_len, d):
        return {
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "length_mm": f"{d['length_mm']:.4f}",
            "width_mm": f"{d['width_mm']:.4f}",
            "thickness_mm": f"{specT:.4f}",
            "length_side_mm": "" if side_len is None else f"{side_len:.4f}",
            "delta_mm": "" if d["delta_mm"] is None else f"{d['delta_mm']:+.4f}",
            "length_px": f"{r0.length_px:.4f}",
            "width_px": f"{r0.width_px:.4f}",
            "thickness_px": f"{r1.width_px:.4f}",
            "length_side_px": (f"{r1.length_px:.4f}"
                               if np.isfinite(r1.length_px) else ""),
            "skew0_deg": ("" if d.get("skew0") is None
                          else f"{d['skew0']:+.4f}"),
            "skew1_deg": ("" if d.get("skew1") is None
                          else f"{d['skew1']:+.4f}"),
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
                # Whichever role is bound to this sensor right now -- so the overlay
                # follows a camera swap without restarting anything.
                role = bottom if bottom.cam is cam else side
                box = None
                if role.last_roi and role.last_w:
                    k = bgr.shape[1] / role.last_w
                    box = [int(v * k) for v in role.last_roi]
                elif role.roi_spec and role.roi_frac:
                    # No successful measurement yet -- grab() may be refusing the
                    # frame -- but a fractional ROI resolves against any size, so
                    # show it anyway. Otherwise a working ROI looks like a broken
                    # button.
                    h, w = bgr.shape[:2]
                    box = [int(role.roi_spec[0] * w), int(role.roi_spec[1] * h),
                           int(role.roi_spec[2] * w), int(role.roi_spec[3] * h)]
                if box:
                    cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]),
                                  (0, 200, 255), 1)
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
                 "ref0": bottom.ref is not None, "ref1": side.ref is not None,
                 "datum0": bottom.datum, "datum1": side.datum,
                 "doff0": bottom.datum_off, "doff1": side.datum_off,
                 "sensor0": bottom.cam.index, "sensor1": side.cam.index,
                 "length_mm": None, "width_mm": None, "thickness_mm": None,
                 "delta_mm": None, "tilt0": None, "tilt1": None,
                 "rms0": None, "rms1": None, "tsd": None, "n": len(hist),
                 "skew0": None, "skew1": None,
                 "roi0": list(bottom.last_roi) if bottom.last_roi else None,
                 "roi1": list(side.last_roi) if side.last_roi else None}
            row = None

            # --- SIDE FIRST: it produces the thickness the bottom camera needs.
            # Its own silhouette plane is the coupon's reference face, so it is measured
            # at offset 0 and needs nothing from the other camera.
            r1 = c1 = k1 = None
            try:
                img1 = side.grab()
                r1, c1, k1 = side.measure(img1, 0.0)
                d["tilt1"], d["rms1"] = float(r1.tilt_deg), float(r1.edge_rms_px)
                d["skew1"] = side.skew()
                d["thickness_mm"] = float(r1.width_px * k1) if k1 else float(r1.width_px)
                if k1:
                    latest["thickness"] = float(r1.width_px * k1)
            except Exception as e:
                d["err1"] = str(e)[:90]

            # --- BOTTOM, using the measured thickness. Never a nominal one: at
            # 775 mm a 1 mm thickness error is 1290 ppm, ten times the length budget,
            # and the result looks perfectly reasonable.
            if a.plane_offset is not None:
                specT = a.plane_offset
            else:
                specT = float(r1.width_px * k1) if (r1 is not None and k1) else None
            try:
                img0 = bottom.grab()
                r0, c0, _ = bottom.measure(img0, 0.0)
                d["tilt0"], d["rms0"] = float(r0.tilt_deg), float(r0.edge_rms_px)
                d["skew0"] = bottom.skew()
                k0 = (c0.mmPerPx(specT)
                      if (bottom.focalPx > 0 and specT is not None) else 0.0)
                d["length_mm"] = float(r0.length_px * k0) if k0 else float(r0.length_px)
                d["width_mm"] = float(r0.width_px * k0) if k0 else float(r0.width_px)
                if k0 and k1:
                    meas_t = float(r1.width_px * k1)
                    side_len = (float(r1.length_px * k1)
                                if np.isfinite(r1.length_px) else None)
                    if side_len is not None:
                        d["delta_mm"] = side_len - d["length_mm"]
                    hist.append(meas_t)
                    row = _row(r0, r1, c0, c1, meas_t, side_len, d)
            except Exception as e:
                d["err0"] = str(e)[:90]

            if d["err1"] and specT is None:
                d["err0"] = d["err0"] or "no thickness from the side view"
            d["fixedPlane"] = a.plane_offset

            if len(hist) > 1:
                arr = np.array(hist)
                d["tsd"] = float(arr.std(ddof=1))
            d["n"] = len(hist)
            # Coupon pressed against opaque fixture: that edge is hidden.
            d["touch0"], d["touch1"] = bottom.touch, side.touch
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
            elif self.path == "/specimens":
                if SR is None:
                    body = {"codes": [], "error": "specimen_registry not available"}
                else:
                    try:
                        body = {"codes": SR.list_specimens()}
                    except Exception as e:
                        body = {"codes": [], "error": str(e)[:150]}
                self._send(json.dumps(body).encode(), "application/json")
            elif self.path == "/settings":
                self._send(json.dumps(settings_doc()).encode(),
                           "application/json")
            elif self.path == "/stream0":
                self._mjpeg(lambda: bottom.cam)     # resolved per frame, so a
            elif self.path == "/stream1":          # swap takes effect live
                self._mjpeg(lambda: side.cam)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/settings":
                nbytes = int(self.headers.get("Content-Length", 0))
                try:
                    req = json.loads(self.rfile.read(nbytes) or b"{}")
                except Exception:
                    return self._send(b"settings: could not read the request")

                # Validate EVERY field before changing ANY of them. A partly
                # applied settings page leaves the station in a state nobody
                # chose and nobody can see.
                wanted, bad = {}, []
                for k, raw in req.items():
                    t = BY_KEY.get(k)
                    if t is None:
                        bad.append(f"{k}: not a setting")
                    else:
                        try:
                            wanted[k] = coerce(t, raw)
                            if k == "new_fields":
                                parse_fields(wanted[k])   # fail here, not at Log
                        except ValueError as e:
                            bad.append(f"{t['label']}: {e}")
                if bad:
                    return self._send(
                        ("NOTHING CHANGED | " + " | ".join(bad)).encode())

                changed, later = [], []
                for k, v in wanted.items():
                    if v == getattr(a, k) and tuned.get(k, v) == v:
                        continue
                    t = BY_KEY[k]
                    if t["scope"] == "restart":
                        later.append(t["label"])
                    else:
                        apply_live(k, v)
                    tuned[k] = v
                    changed.append(f"{k}={'(blank)' if v is None else v}")
                if not changed:
                    return self._send(b"settings unchanged")
                save_state()
                hist.clear()
                out = f"applied: {', '.join(changed)} | saved to {a.state}"
                if later:
                    out += f" | needs a restart: {', '.join(later)}"
                return self._send(out.encode())

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
                    # Calibrate at the same offset this view measures at.
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
                # Profile BOTH viewpoints.
                out, rows = [], []
                # Each view profiles INDEPENDENTLY.
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
                        if role.datum:
                            # Small axis against the datum; length as a bonus.
                            r, ex = role.measure(img, off)[0], role.extras
                            mmpx = c.mmPerPx(off) if role.focalPx > 0 else 0.0
                            sp = C.chords_from_values(ex["pos"], ex["small_px"], mmpx,
                                                      a.chords, band, a.trim, small,
                                                      r.tilt_deg)
                            try:
                                lp, _ = C.measure_chords(img, c, off, a.chords, band,
                                                         a.trim, a.order)
                            except Exception:
                                lp = None
                        else:
                            lp, sp = C.measure_chords(img, c, off, a.chords, band,
                                                      a.trim, a.order)
                    except Exception as e:
                        out.append(f"{label}: {str(e)[:80]}")
                        continue
                    cal = role.focalPx > 0
                    u = "mm" if cal else "px"
                    for pr, nm in ((lp, "length"), (sp, small)):
                        if pr is None:
                            continue
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

            if self.path == "/reference":
                n = int(self.headers.get("Content-Length", 0))
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                    which = int(req.get("camera", 0))
                    action = req.get("action", "capture")
                except Exception as e:
                    return self._send(f"bad request: {e}".encode())
                role = bottom if which == 0 else side
                if action == "clear":
                    role.clear_reference()
                    hist.clear()
                    return self._send(f"{role.name}: reference cleared".encode())
                try:
                    ref, clipped = role.capture_reference(a.ref_frames)
                except Exception as e:
                    return self._send(f"reference failed: {e}".encode())
                hist.clear()
                msg = (f"{role.name}: reference from {a.ref_frames} frames | "
                       f"background {ref.bright:.0f} DN | "
                       f"{ref.masked_frac*100:.1f}% of frame opaque")
                if clipped > 0.01:
                    msg += (f" | WARNING {clipped:.2f}% saturated - lower the "
                            f"exposure and recapture; a clipped reference cannot "
                            f"divide correctly")
                msg += (" | recalibrate this view: edge positions have moved to "
                        "their true place")
                return self._send(msg.encode())

            if self.path == "/roi":
                n = int(self.headers.get("Content-Length", 0))
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                    which = int(req.get("camera", 0))
                    spec = req.get("roi", "")
                except Exception as e:
                    return self._send(f"bad request: {e}".encode())
                role = bottom if which == 0 else side
                try:
                    kind = role.set_roi(spec)
                except Exception as e:
                    return self._send(f"bad ROI: {e}".encode())
                hist.clear()
                save_state()
                return self._send(
                    (f"{role.name}: ROI now {kind}"
                     + ("" if kind == "auto" else f" {role.roi_spec}")
                     + " - watch the overlay and the status rows").encode())

            if self.path == "/swap":
                swapped["v"] = not swapped["v"]
                # A reference is a picture of what one sensor saw from one place.
                bottom.clear_reference()
                side.clear_reference()
                bind()
                save_state()
                # The thickness history describes the old pairing; keeping it would
                # blend two different geometries into one spread figure.
                hist.clear()
                latest["thickness"] = None
                return self._send(
                    (f"swapped: bottom is now sensor "
                     f"{bottom.cam.index}, side is sensor {side.cam.index}. "
                     f"Recalibrate both -- the calibrations belong to the "
                     f"viewpoints, and the viewpoints have changed sensors."
                     ).encode())

            if self.path == "/log":
                nbytes = int(self.headers.get("Content-Length", 0))
                try:
                    req = json.loads(self.rfile.read(nbytes) or b"{}")
                except Exception:
                    req = {}
                sid = str(req.get("specimen_id", "")).strip().upper()
                mode = str(req.get("mode", "existing")).strip().lower()

                row = snap["row"]
                if not row:
                    # snap["row"] only exists once BOTH views are calibrated, so
                    # this also guarantees the values below are millimetres.
                    return self._send(b"no complete measurement to log - both "
                                      b"views must be calibrated")

                out = []
                if mode == "new":
                    # Mint the code BEFORE writing the CSV, or the local row gets
                    # the series prefix ("PS-V4") as its ID and never matches the
                    # specimen that ends up in the app.
                    #
                    # The database's uniqueness on specimen_id is the arbiter, not
                    # our copy of the list -- the press or another station can take
                    # a number a second before we do. So: propose, insert, and on
                    # rejection try the next one. A consequence worth knowing: a
                    # NEW specimen cannot be created offline. The reading is still
                    # written to the CSV below, so nothing is lost -- you just have
                    # to attach it to a code later.
                    if SR is None:
                        out.append("specimen_registry not available - no new ID")
                        sid = ""
                    else:
                        try:
                            # The dashboard reads the dimensions off the
                            # specimens row itself, so put them in the insert
                            # that creates it. Doing it now, while we are
                            # already writing the row, avoids ever having to
                            # UPDATE a specimen -- the history in
                            # identifier_measurements stays append-only and
                            # nothing is overwritten. Anything typed into
                            # "new specimen fields" wins, so a bad reading can
                            # still be overridden by hand.
                            born = {"length_mm": round(float(row["length_mm"]), 2),
                                    "width_mm": round(float(row["width_mm"]), 2),
                                    "thickness_mm": round(float(row["thickness_mm"]), 3)}
                            born.update(parse_fields(a.new_fields))
                            sid = SR.claim_next(sid or "PS-V4", **born)
                            out.append(f"created {sid} with "
                                       + ", ".join(sorted(born)))
                        except Exception as e:
                            out.append(f"could not create a specimen "
                                       f"({str(e)[:220]}) - logged WITHOUT an ID")
                            sid = ""
                row = dict(row, specimen_id=sid)

                if a.log:
                    fresh = (not os.path.exists(a.log)
                             or os.path.getsize(a.log) == 0)
                    with open(a.log, "a", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
                        if fresh:
                            w.writeheader()
                        w.writerow(row)
                    out.append(f"logged to {a.log}")
                else:
                    out.append("no --log file given")

                if not sid:
                    out.append("no specimen ID - not sent to the app")
                elif send_measurement is None:
                    out.append("identifier_push not available - not sent")
                else:
                    # Carry the provenance the app's table cannot hold yet, so a
                    # row can still be judged later: the two cameras' independent
                    # lengths, the fit quality, and the thickness spread.
                    note = (f"delta {row['delta_mm']} mm | "
                            f"rms {row['rms0_px']}/{row['rms1_px']} px | "
                            f"tilt {row['tilt0_deg']}/{row['tilt1_deg']} deg")
                    if a.note:
                        note = f"{a.note} | {note}"
                    try:
                        f = send_measurement(
                            sid,
                            length_mm=float(row["length_mm"]),
                            width_mm=float(row["width_mm"]),
                            thickness_mm=float(row["thickness_mm"]),
                            notes=note)
                        # send_measurement returns the outbox file whether or
                        # not the app took it. The file still being there is
                        # the only honest signal that it did not -- reporting
                        # "queued" either way hides a table that is refusing
                        # every row.
                        if f.exists():
                            waiting = len(list(f.parent.glob("*.json")))
                            out.append(f"NOT accepted by the app - {waiting} "
                                       f"waiting in outbox/ (see the terminal "
                                       f"for the reason)")
                        else:
                            out.append("measurement accepted by the app")
                    except Exception as e:
                        out.append(f"PUSH FAILED: {str(e)[:120]}")
                return self._send(" | ".join(out).encode())
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