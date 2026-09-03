#!/usr/bin/env python3
"""
pi_capture.py -- metrology-grade frames from a Raspberry Pi HQ Camera.

Captures RAW Bayer and extracts the green plane. Saves a 16-bit PNG that
gauge.py reads directly.

    python3 pi_capture.py --info                  what the sensor offers
    python3 pi_capture.py -o shot.png             one frame
    python3 pi_capture.py -o shot.png --exposure 8000 --gain 1.0
    python3 pi_capture.py -o run --frames 20      run_000.png .. run_019.png
"""

import argparse
import sys
import time

import numpy as np

import gauge as G

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("picamera2 not found.  sudo apt install -y python3-picamera2")

try:
    import cv2
except ImportError:
    sys.exit("opencv not found.  sudo apt install -y python3-opencv")


# ----------------------------------------------------------------- bayer ---

def green_plane(raw: np.ndarray, fmt: str) -> np.ndarray:
    """Average the two green sites of every Bayer tile.

    The format string names the tile order top-left, top-right, bottom-left,
    bottom-right -- e.g. SRGGB12 means R G / G B. The two greens are always on
    one diagonal or the other, so only two cases exist. Detect it rather than
    assume, because the order flips when hflip or vflip is set.
    """
    order = fmt.lstrip("S")[:4]          # "RGGB", "BGGR", "GRBG", "GBRG"
    if order in ("RGGB", "BGGR"):        # greens at (0,1) and (1,0)
        g1, g2 = raw[0::2, 1::2], raw[1::2, 0::2]
    elif order in ("GRBG", "GBRG"):      # greens at (0,0) and (1,1)
        g1, g2 = raw[0::2, 0::2], raw[1::2, 1::2]
    else:
        raise ValueError(f"unrecognised Bayer order in {fmt!r}")
    n = min(g1.shape[0], g2.shape[0]), min(g1.shape[1], g2.shape[1])
    return 0.5 * (g1[:n[0], :n[1]].astype(np.float32) +
                  g2[:n[0], :n[1]].astype(np.float32))


# ----------------------------------------------------------------- camera --

def open_camera(args):
    picam2 = Picamera2()

    # Ask for UNPACKED raw. The default on the Pi is CSI2P, which crams two
    # 12-bit pixels into three bytes and would need manual unpacking.
    want = (args.width, args.height)
    fmt = None
    for m in picam2.sensor_modes:
        if tuple(m["size"]) == want:
            fmt = m["format"].format if hasattr(m["format"], "format") else str(m["format"])
            break
    if fmt is None:
        print(f"# no sensor mode at {want}; using the largest available")
        m = max(picam2.sensor_modes, key=lambda m: m["size"][0] * m["size"][1])
        want = tuple(m["size"])
        fmt = m["format"].format if hasattr(m["format"], "format") else str(m["format"])
    unpacked = fmt.replace("_CSI2P", "")

    if getattr(args, "live", False):
        cfg = picam2.create_video_configuration(
            main={"size": (800, 600), "format": "RGB888"},
            raw={"size": want, "format": unpacked},
            buffer_count=3,
        )
    else:
        cfg = picam2.create_still_configuration(
            raw={"size": want, "format": unpacked},
            buffer_count=2,
        )
    picam2.configure(cfg)

    exposure, gain = args.exposure, args.gain

    if args.auto_once:
        picam2.set_controls({"AeEnable": True, "AwbEnable": False})
        picam2.start()
        time.sleep(2.5)                        # let the AE loop converge
        md = picam2.capture_metadata()
        exposure = int(md.get("ExposureTime", exposure))
        gain = float(md.get("AnalogueGain", gain))
        print(f"# auto-exposure chose {exposure} us at gain {gain:.2f} -- now pinned")
        print(f"#   reuse it directly next time:  --exposure {exposure} --gain {gain:.2f}")
        picam2.stop()

    picam2.set_controls({
        "AeEnable": False,             # auto exposure: brightness would drift frame to frame
        "AwbEnable": False,            # auto white balance: per-channel gains chasing the scene
        "ExposureTime": exposure,      # microseconds
        "AnalogueGain": gain,          # 1.0 = minimum = lowest read noise
    })
    picam2.start()
    time.sleep(1.5)                    # let the sensor settle on the new exposure
    rawcfg = picam2.camera_configuration()["raw"]
    return picam2, rawcfg["format"], tuple(rawcfg["size"]), exposure, gain


def capture(picam2, fmt, size) -> np.ndarray:
    """Grab one raw frame and return the green plane as float32."""
    w, h = size
    raw = picam2.capture_array("raw")

    if raw.dtype == np.uint8:
        raw = np.ascontiguousarray(raw).view(np.uint16)   # bytes -> 16-bit pixels
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise RuntimeError(f"unexpected raw shape {raw.shape} for format {fmt}")

    if raw.shape[1] < w or raw.shape[0] < h:
        raise RuntimeError(
            f"raw buffer {raw.shape} smaller than the configured {h}x{w}")
    raw = raw[:h, :w]                                     # drop the stride padding

    return green_plane(raw, fmt)


def infer_bits(g: np.ndarray) -> int:
    """Bit depth.
    """
    m = float(g.max())
    for b in (8, 10, 12, 14, 16):
        if m <= 2 ** b - 1:
            return b
    return 16


def report(g: np.ndarray, bits: int) -> None:
    """The three numbers worth checking before trusting any measurement."""
    full = float(2 ** bits - 1)
    lo, hi = float(g.min()), float(g.max())
    p5, p95 = np.percentile(g, 5), np.percentile(g, 95)
    clip = float((g >= full * 0.995).mean() * 100)
    print(f"  {g.shape[1]} x {g.shape[0]} green plane, {bits}-bit")
    print(f"  range {lo:.0f}..{hi:.0f}   5th/95th {p5:.0f}/{p95:.0f}"
          f"   contrast {p95-p5:.0f}")
    print(f"  saturated {clip:.2f}%")
    if clip > 0.01:
        print("  >> background is clipping. Both edges walk INWARD and the coupon")
        print("     reads UNDERSIZE, silently, with excellent repeatability.")
        print("     Lower --exposure.")
    elif p95 < full * 0.5:
        print("  >> dim. Raise --exposure for more edge contrast; aim the bright")
        print("     field at roughly 80% of full scale.")




# ------------------------------------------------------------------- live --

LIVE_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>PaperSteel capture</title><style>
body{background:#1a1a1a;color:#e8e8e8;font:14px system-ui,sans-serif;margin:0;padding:16px}
.wrap{display:flex;gap:16px;flex-wrap:wrap}
img{border:1px solid #444;max-width:800px;width:100%}
.panel{min-width:290px;flex:1}
button{font:600 15px system-ui;padding:11px 22px;border:0;border-radius:6px;
 background:#2d7d5a;color:#fff;cursor:pointer}
button:hover{background:#379a6e} button:disabled{background:#444;cursor:wait}
table{border-collapse:collapse;width:100%;margin:10px 0}
td{padding:3px 8px;border-bottom:1px solid #333} td:first-child{color:#999}
td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.warn{color:#e8a33d} .bad{color:#e06c5a} .ok{color:#6ec48f}
pre{background:#111;padding:10px;border-radius:5px;white-space:pre-wrap;font-size:12px}
h3{margin:14px 0 4px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#888}
</style></head><body>
<div class="wrap">
 <div><img id="v" src="/stream"></div>
 <div class="panel">
  <button id="b" onclick="grab()">Capture &amp; measure</button>
  <h3>Live</h3><table id="live"></table>
  <h3>Last capture</h3><pre id="out">nothing yet</pre>
 </div>
</div>
<script>
function row(k,v,c){return '<tr><td>'+k+'</td><td class="'+(c||'')+'">'+v+'</td></tr>'}
async function poll(){
 try{const s=await (await fetch('/stats')).json();
  let cls = s.clipped>0.01?'bad':(s.p95frac>0.9?'warn':'ok');
  document.getElementById('live').innerHTML =
    row('bright field', s.p95+' / '+s.full, cls)+
    row('contrast', s.contrast)+
    row('clipped', s.clipped.toFixed(2)+'%', s.clipped>0.01?'bad':'')+
    row('focus (edge px)', s.edge===null?'no edge':s.edge.toFixed(2),
        s.edge!==null&&s.edge<3?'ok':'warn')+
    row('exposure', s.exposure+' us');
 }catch(e){}
 setTimeout(poll,700);
}
async function grab(){
 const b=document.getElementById('b'); b.disabled=true; b.textContent='Capturing...';
 try{ document.getElementById('out').textContent =
        (await (await fetch('/capture',{method:'POST'})).json()).text; }
 catch(e){ document.getElementById('out').textContent='failed: '+e; }
 b.disabled=false; b.textContent='Capture & measure';
}
poll();
</script></body></html>"""


def live_server(picam2, fmt, size, exposure, args):
    """Serve an MJPEG preview with a Capture button.
    """
    import http.server, io, json, socket, threading

    lock = threading.Lock()          # only one thread may touch the camera
    state = {"jpeg": None, "stats": {}}
    gcfg = G.Config()

    def preview_loop():
        while True:
            with lock:
                rgb = picam2.capture_array("main")
            grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            p5, p95 = np.percentile(grey, 5), np.percentile(grey, 95)
            try:
                gw = G.grad_width(cv2.GaussianBlur(grey, (0, 0), 1.0), gcfg)
            except Exception:
                gw = None
            ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            state["jpeg"] = buf.tobytes() if ok else None
            # Cast every value to a plain Python type: numpy scalars are not
            # JSON serializable and would 500 the whole stats endpoint.
            state["stats"] = {
                "p95": int(p95), "full": 255, "p95frac": float(p95) / 255.0,
                "contrast": int(p95 - p5),
                "clipped": float((grey >= 254).mean() * 100),
                "edge": None if gw is None else float(gw),
                "exposure": int(exposure),
            }
            time.sleep(0.05)

    def do_capture():
        with lock:
            g = capture(picam2, fmt, size)
        bits = infer_bits(g)
        path = time.strftime("cap_%Y%m%d_%H%M%S.png")
        cv2.imwrite(path, np.clip(g, 0, 65535).astype(np.uint16))

        full = 2 ** bits - 1
        p5, p95 = np.percentile(g, 5), np.percentile(g, 95)
        clip = (g >= full * 0.995).mean() * 100
        out = [f"{path}   {g.shape[1]} x {g.shape[0]}, {bits}-bit",
               f"bright field {p95:.0f}/{full}   contrast {p95-p5:.0f}"
               f"   clipped {clip:.2f}%"]
        if clip > 0.01:
            out.append("!! clipping - reads UNDERSIZE. Lower the exposure.")

        img = g * (255.0 / full)                      # gauge works in 0-255 float
        try:
            c = G.Config()
            c.roi = G.auto_roi(img)
            c.specT = args.thickness
            if args.eff_dist:
                c.effDist = args.eff_dist
            gw = G.grad_width(cv2.GaussianBlur(
                img[c.roi[1]:c.roi[3], c.roi[0]:c.roi[2]], (0, 0), 1.0), c)
            out.append(f"ROI {c.roi}   edge width {gw:.2f} px")
            if args.focal_px:
                c.focalPx = args.focal_px
                r = G.measure(img, c, args.thickness)
                out.append(f"LENGTH {r.length_mm:9.4f} mm   ({r.length_px:.3f} px)")
                out.append(f"WIDTH  {r.width_mm:9.4f} mm   ({r.width_px:.3f} px)")
                out.append(f"scale {r.um_per_px:.4f} um/px   tilt {r.tilt_deg:+.3f} deg"
                           f"   edge RMS {r.edge_rms_px:.4f} px")
            else:
                out.append("no --focal-px given, so no mm. Calibrate with:")
                out.append(f"  python3 gauge.py --image {path} --auto-roi "
                           f"--thickness {args.thickness} --calibrate <known mm>")
        except Exception as e:
            out.append(f"measurement failed: {e}")
        return "\n".join(out)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                body = LIVE_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stats":
                body = json.dumps(state["stats"]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        j = state["jpeg"]
                        if j:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: " +
                                             str(len(j)).encode() + b"\r\n\r\n" +
                                             j + b"\r\n")
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/capture":
                return self.send_error(404)
            body = json.dumps({"text": do_capture()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    threading.Thread(target=preview_loop, daemon=True).start()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), H)
    print(f"\n#   open  http://{ip}:{args.port}   (Ctrl-C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("# stopping")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default="frame.png")
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--exposure", type=int, default=10000,
                   help="microseconds. NOT a camera default -- just a starting "
                        "guess. Use --auto-once or --sweep to find yours.")
    p.add_argument("--auto-once", action="store_true",
                   help="let auto-exposure pick a value, print it, then pin it")
    p.add_argument("--sweep", action="store_true",
                   help="try a range of exposures and report contrast for each")
    p.add_argument("--gain", type=float, default=1.0, help="analogue gain, 1.0 = min")
    p.add_argument("--width", type=int, default=4056)
    p.add_argument("--height", type=int, default=3040)
    p.add_argument("--info", action="store_true", help="list sensor modes and exit")
    p.add_argument("--live", action="store_true",
                   help="serve a live preview with a Capture button")
    p.add_argument("--port", type=int, default=8080, help="port for --live")
    p.add_argument("--thickness", type=float, default=5.5,
                   help="specT of the coupon, mm -- used when measuring a capture")
    p.add_argument("--eff-dist", type=float, help="A, platform-to-pinhole, mm")
    p.add_argument("--focal-px", type=float,
                   help="calibration; without it captures report pixels only")
    a = p.parse_args()

    if a.info:
        pc = Picamera2()
        for m in pc.sensor_modes:
            print(f"  {m['size']}  {m['format']}  bit_depth={m.get('bit_depth')}"
                  f"  fps<={m.get('fps')}")
        pc.close()
        return

    picam2, fmt, size, exposure, gain = open_camera(a)
    print(f"# raw format {fmt}, {size[0]}x{size[1]}, "
          f"exposure {exposure} us, gain {gain}")

    if a.live:
        live_server(picam2, fmt, size, exposure, a)
        picam2.stop(); picam2.close()
        return

    if a.sweep:
        print(f"\n  {'exposure':>10}{'95th pct':>11}{'contrast':>11}{'clipped':>10}  verdict")
        best = None
        for e in [int(exposure * k) for k in (0.25, 0.5, 1, 2, 4, 8)]:
            picam2.set_controls({"ExposureTime": e})
            time.sleep(0.8)
            capture(picam2, fmt, size)               # discard: sensor still settling
            g = capture(picam2, fmt, size)
            bits = infer_bits(g); full = 2 ** bits - 1
            p5, p95 = np.percentile(g, 5), np.percentile(g, 95)
            clip = (g >= full * 0.995).mean() * 100
            if clip > 0.01:            v = "CLIPPING"
            elif p95 > full * 0.9:     v = "close to clipping"
            elif p95 < full * 0.4:     v = "dim"
            else:
                v = "good"
                if best is None or p95 > best[1]: best = (e, p95)
            print(f"  {e:>9}us{p95:>11.0f}{p95-p5:>11.0f}{clip:>9.2f}%  {v}")
        if best:
            print(f"\n  -> use --exposure {best[0]}")
        else:
            print("\n  -> nothing landed in range; widen the sweep or check the backlight")
        picam2.stop(); picam2.close()
        return

    bits = None
    for i in range(a.frames):
        g = capture(picam2, fmt, size)
        if bits is None:
            bits = infer_bits(g)
            print(f"# data is {bits}-bit (inferred from the values, not the format name)")
        path = a.out if a.frames == 1 else f"{a.out}_{i:03d}.png"
        # 16-bit PNG: keeps all 12 bits. gauge.py normalises on load, so the
        # extra precision survives into the float the edge finder works on.
        cv2.imwrite(path, np.clip(g, 0, 65535).astype(np.uint16))
        print(f"# wrote {path}")
        if i == 0:
            report(g, bits)

    picam2.stop()
    picam2.close()


if __name__ == "__main__":
    main()