"""One persistent camera owner, live preview and a local terminal calibration client."""

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


class StablePair:
    """A short consecutive stable observation, never an arbitrary historical pair."""

    def __init__(self, min_frames=5, max_age_s=2.0, max_spread_px=2.0):
        self.frames = deque(maxlen=min_frames)
        self.min_frames = min_frames
        self.max_age_s = max_age_s
        self.max_spread_px = max_spread_px
        self.ready = None
        self.last_used = None
        self.reason = "Waiting for both markers"

    def update(self, timestamp, frame_number, separation=None, reason=None):
        if separation is None:
            self.frames.clear()
            self.reason = reason or "Markers not visible"
            return
        if self.frames and (
            frame_number <= self.frames[-1][1] or timestamp - self.frames[-1][0] > 0.15
        ):
            self.frames.clear()
        self.frames.append((timestamp, frame_number, float(separation)))
        if len(self.frames) < self.min_frames:
            self.reason = (
                f"Both visible: holding {len(self.frames)}/{self.min_frames} frames"
            )
            return
        values = [r[2] for r in self.frames]
        spread = float(np.ptp(values))
        if spread > self.max_spread_px:
            self.ready = None
            self.reason = f"Hold steady: separation varies by {spread:.2f} px"
            return
        self.ready = {
            "monotonic_s": timestamp,
            "frame_number": frame_number,
            "frame_numbers": [r[1] for r in self.frames],
            "marker_separation_px": float(np.median(values)),
            "separation_p95_p05": float(
                np.quantile(values, 0.95) - np.quantile(values, 0.05)
            ),
            "window_span_s": timestamp - self.frames[0][0],
            "valid_frames": len(values),
        }
        self.reason = "READY: enter the measured opening in your terminal"

    def snapshot(self, now):
        if self.ready is None or now - self.ready["monotonic_s"] > self.max_age_s:
            raise ValueError(
                "No fresh stable pair. Hold both markers visible, wait for READY, then submit again."
            )
        if (
            self.last_used is not None
            and min(self.ready["frame_numbers"]) <= self.last_used
        ):
            raise ValueError(
                "This observation was already saved. Wait for five fresh frames."
            )
        return dict(self.ready, age_s=now - self.ready["monotonic_s"])


def identity(config):
    return {
        "width_method": "image_calibrated",
        "resolution": config["resolution"],
        "rgb_serial": config["rgb_serial"],
        "markers": {k: config["markers"][k] for k in ("dictionary", "ids", "size_m")},
    }


def append_sample(path, config, sample, opening_mm):
    if (
        not np.isfinite(opening_mm)
        or opening_mm < 0
        or opening_mm > config["markers"]["width_range_m"][1] * 1000
    ):
        raise ValueError("Opening must be finite and within configured physical travel")
    path = Path(path)
    doc = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "calibration_identity": identity(config),
            "marker_separation_px": [],
            "jaw_opening_m": [],
            "captures": [],
        }
    )
    if (
        doc.get("calibration_identity") != identity(config)
        or "marker_separation_px" not in doc
    ):
        raise ValueError("Sample file does not match this calibration setup")
    capture = sample | {
        "measured_jaw_opening_m": opening_mm / 1000,
        "unix_s": time.time(),
        "separation_units": "pixels",
        "reprojection_p95_px": -1.0,
        "source": "live_terminal",
    }
    doc["marker_separation_px"].append(capture["marker_separation_px"])
    doc["jaw_opening_m"].append(opening_mm / 1000)
    doc["captures"].append(capture)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.flush()
        import os

        os.fsync(f.fileno())
    tmp.replace(path)
    return {
        "saved": True,
        "sample_count": len(doc["captures"]),
        "opening_mm": opening_mm,
        "separation_px": capture["marker_separation_px"],
        "frames": capture["valid_frames"],
        "observation_age_s": capture["age_s"],
        "samples_path": str(path),
    }


def serve(config, samples_path, port=8080, control_port=8081):
    from .hardware import intrinsics, sdk
    from .viewer import Viewer

    if config["markers"].get("width_method") != "image_calibrated":
        raise ValueError("Live calibration requires image_calibrated width mode")
    guard = threading.Lock()
    stop = threading.Event()
    state = StablePair()
    last_result = {"message": "No samples saved in this session"}
    latest_frame = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/frame.png":
                import cv2

                with guard:
                    rgb = latest_frame.get("rgb")
                if rgb is None:
                    self.send_error(503, "Waiting for camera")
                    return
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    self.send_error(500)
                    return
                body = encoded.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/status":
                self.send_error(404)
                return
            with guard:
                data = {"status": state.reason, "last_result": dict(last_result)}
                try:
                    data["ready"] = state.snapshot(time.monotonic())
                except ValueError:
                    data["ready"] = None
            self.respond(200, data)

        def do_POST(self):
            if self.path != "/sample":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                opening = float(payload["opening_mm"])
                with guard:
                    sample = state.snapshot(time.monotonic())
                    result = append_sample(samples_path, config, sample, opening)
                    state.last_used = sample["frame_number"]
                    last_result.clear()
                    last_result.update(result)
                print(json.dumps(result), flush=True)
                self.respond(200, result)
            except (ValueError, KeyError, TypeError, OSError) as e:
                self.respond(400, {"error": str(e)})

        def respond(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    http = ThreadingHTTPServer(("127.0.0.1", control_port), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    rs = sdk()
    pipe = rs.pipeline()
    c = rs.config()
    c.enable_device(config["rgb_serial"])
    w, h = config["resolution"]
    c.enable_stream(rs.stream.color, w, h, rs.format.rgb8, config["fps"])
    viewer = None
    started = False
    try:
        profile = pipe.start(c)
        started = True
        viewer = Viewer(config, intrinsics(profile.get_stream(rs.stream.color)), port)
        feedback = viewer.server.gui.add_markdown("Waiting for a stable marker pair")
        print(
            f"Live width calibration ready. Terminal: uv run --locked yumi width-console --control-port {control_port}",
            flush=True,
        )
        last_frame = -1
        while not stop.is_set():
            frame = pipe.wait_for_frames(3000).get_color_frame()
            number = frame.get_frame_number()
            if number <= last_frame:
                continue
            last_frame = number
            rgb = np.asanyarray(frame.get_data())
            now = time.monotonic()
            with guard:
                latest_frame["rgb"] = rgb.copy()
            try:
                separation, _, _ = viewer.markers.image_separation(rgb)
                with guard:
                    state.update(now, number, separation)
            except ValueError as e:
                with guard:
                    state.update(now, number, reason=str(e))
            viewer.update({"rgb": rgb})
            with guard:
                try:
                    snap = state.snapshot(time.monotonic())
                    current = f"READY — {snap['valid_frames']} stable frames; observation age {snap['age_s']:.2f}s. Hold the measured gap while typing."
                except ValueError:
                    current = state.reason + " — no fresh unsaved observation"
                feedback.content = current + "\n\nLast save: " + json.dumps(last_result)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)
        if started:
            pipe.stop()
        if viewer:
            viewer.close()


def console(control_port=8081):
    base = f"http://127.0.0.1:{control_port}"
    print(
        "Keep the browser preview open. Hold the measured jaw gap steady.\n"
        "Enter millimeters (e.g. 30) to save a fresh stable observation.\n"
        "Enter s for status; q to leave this terminal (server stays running).",
        flush=True,
    )
    while True:
        try:
            text = input("Opening in mm [s/q]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() == "q":
            return
        try:
            if text.lower() == "s":
                request = Request(base + "/status")
            else:
                opening = float(text)
                if not np.isfinite(opening) or opening < 0:
                    raise ValueError("Enter a nonnegative measured opening")
                request = Request(
                    base + "/sample",
                    data=json.dumps({"opening_mm": opening}).encode(),
                    headers={"Content-Type": "application/json"},
                )
            with urlopen(request, timeout=5) as response:
                result = json.load(response)
            if result.get("saved"):
                print(
                    f"SAVED #{result['sample_count']}: {result['opening_mm']:g} mm -> {result['separation_px']:.2f} px, {result['frames']} frames, age {result['observation_age_s']:.2f}s",
                    flush=True,
                )
            else:
                print(json.dumps(result, indent=2), flush=True)
        except HTTPError as e:
            print("NOT SAVED: " + json.load(e).get("error", str(e)), flush=True)
        except (ValueError, URLError, TimeoutError) as e:
            print(f"NOT SAVED: {e}", flush=True)
