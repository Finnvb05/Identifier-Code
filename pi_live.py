#!/usr/bin/env python3
"""
pi_live.py -- live camera view in the browser, with a button to save a photo.

    python3 pi_live.py
    python3 pi_live.py --port 8080 --dir shots

Open the URL it prints. Press Save. That is all it does.
"""

import argparse
import http.server
import socket
import threading
import time

import cv2
from picamera2 import Picamera2

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>camera</title>
<style>body{background:#1b1b1b;color:#eee;font:15px system-ui;margin:0;padding:16px;
text-align:center}img{max-width:100%;border:1px solid #444}
button{font:600 16px system-ui;padding:12px 30px;margin:14px;border:0;border-radius:6px;
background:#2d7d5a;color:#fff;cursor:pointer}button:hover{background:#379a6e}
#m{color:#9c9;min-height:22px}</style></head><body>
<img src="/stream"><br><button onclick="save()">Save photo</button>
<div id="m"></div><script>
async function save(){document.getElementById('m').textContent='saving...';
 const r=await fetch('/save',{method:'POST'});
 document.getElementById('m').textContent=await r.text();}
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--dir", default=".", help="where to save photos")
    p.add_argument("--width", type=int, default=4056, help="saved image width")
    p.add_argument("--height", type=int, default=3040, help="saved image height")
    p.add_argument("--exposure", type=int, help="microseconds; omit for auto")
    a = p.parse_args()

    picam2 = Picamera2()
    # main = the full-size frame that gets saved; lores = the small one that
    # gets streamed. Both come from the same configuration, so pressing Save
    # never reconfigures the camera or disturbs the view.
    picam2.configure(picam2.create_video_configuration(
        main={"size": (a.width, a.height), "format": "RGB888"},
        lores={"size": (800, 600), "format": "YUV420"},
    ))
    if a.exposure:
        picam2.set_controls({"AeEnable": False, "ExposureTime": a.exposure})
    picam2.start()
    time.sleep(1.5)

    latest = {"jpeg": None}

    def stream_loop():
        while True:
            yuv = picam2.capture_array("lores")
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                latest["jpeg"] = buf.tobytes()
            time.sleep(0.05)

    threading.Thread(target=stream_loop, daemon=True).start()

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
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        j = latest["jpeg"]
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
            if self.path != "/save":
                return self.send_error(404)
            name = f"{a.dir}/photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            picam2.capture_file(name)
            print(f"saved {name}")
            self._send(f"saved {name}".encode())

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()

    print(f"\n  open  http://{ip}:{a.port}   (Ctrl-C to stop)\n")
    try:
        http.server.ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        picam2.close()


if __name__ == "__main__":
    main()