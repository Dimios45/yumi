"""Persistent device session with an isolated official LeRobot v3 writer."""

import json
import pickle
import queue
import select
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path


def send_packet(pipe, value):
    data = pickle.dumps(value, protocol=4)
    pipe.write(struct.pack("!Q", len(data)))
    pipe.write(data)
    pipe.flush()


class Session:
    def __init__(self, output, episodes, seconds, warmup, task, repo_id):
        if episodes < 1 or seconds <= 0 or warmup < 0 or not task.strip():
            raise ValueError(
                "Require positive episodes/seconds, nonnegative warmup and a task"
            )
        self.root = Path(output).resolve()
        if self.root.exists():
            raise FileExistsError(f"Use a new dataset directory: {self.root}")
        self.episodes, self.seconds, self.warmup = episodes, seconds, warmup
        self.task, self.repo_id = task, repo_id
        self.active = False
        self.lock = threading.Lock()
        self.q = queue.Queue(maxsize=512)
        self.failure = None
        self.replies = queue.Queue()
        self.process = None
        self.thread = None
        self.saved = 0
        self.closed = False
        self.meta = None

    def setup(self, meta):
        self.meta = meta
        project = Path(__file__).resolve().parents[2] / "exporter"
        python = project / ".venv/bin/python"
        if not python.exists():
            raise RuntimeError(
                "Install the writer first: uv sync --project exporter --locked"
            )
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root.with_name(self.root.name + ".writer.log")
        self.writer_log = self.log_path.open("x")
        self.process = subprocess.Popen(
            [
                str(python),
                str(project / "session_writer.py"),
                "--root",
                str(self.root),
                "--repo-id",
                self.repo_id,
                "--task",
                self.task,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
            stderr=self.writer_log,
        )
        self.writer_log.close()
        threading.Thread(target=self._read_replies, daemon=True).start()
        send_packet(self.process.stdin, {"command": "setup", "metadata": meta})
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _read_replies(self):
        for line in self.process.stdout:
            try:
                self.replies.put(json.loads(line))
            except ValueError:
                self.failure = RuntimeError(f"Unexpected writer output: {line!r}")
        if not self.closed and self.process.poll() not in (None, 0):
            self.failure = RuntimeError("LeRobot writer exited; see error above")

    def _write(self):
        import numpy as np

        from .geometry import pose_matrix, state
        from .markers import Markers

        detector = Markers(self.meta["config"]["markers"], self.meta["rgb_intrinsics"])
        ext = np.asarray(self.meta["config"]["T_tracker_tcp"])
        pose = None
        try:
            while True:
                packet = self.q.get()
                try:
                    if packet is None:
                        return
                    if packet.get("command") == "begin":
                        pose = None
                    item = packet.get("sample", {})
                    if item.get("kind") == "pose":
                        pose = item
                    if item.get("kind") == "image":
                        width, width_ok = 0.0, False
                        try:
                            width, _, _ = detector.width(item["rgb"])
                            width_ok = True
                        except ValueError as e:
                            item["width_error"] = str(e)
                        item["width_measurement"] = {
                            "valid": width_ok,
                            "opening_m": width if width_ok else None,
                        }
                        pose_ok = (
                            pose is not None
                            and abs(item["device_s"] - pose["device_s"]) < 0.05
                        )
                        matrix = (
                            pose_matrix(pose["position"], pose["quaternion_xyzw"]) @ ext
                            if pose is not None
                            else np.eye(4)
                        )
                        item["provisional_state"] = state(matrix, width).astype(
                            np.float32
                        )
                        item["quality"] = np.array(
                            [
                                pose["tracker_confidence"] if pose else 0,
                                float(pose_ok),
                                float(width_ok),
                                item["device_s"] - pose["device_s"] if pose else 0.0,
                            ],
                            dtype=np.float32,
                        )
                    send_packet(self.process.stdin, packet)
                finally:
                    self.q.task_done()
        except Exception as e:  # noqa: BLE001 — report background writer failures
            self.failure = e

    def accept(self, item):
        with self.lock:
            if self.active:
                try:
                    self.q.put_nowait({"sample": item})
                except queue.Full as e:
                    self.failure = RuntimeError(
                        "LeRobot writer cannot keep up; recording aborted"
                    )
                    raise self.failure from e

    def check(self):
        if self.failure:
            raise RuntimeError(
                f"Session writer failed: {self.failure}; see {self.log_path}"
            )
        if self.thread and not self.thread.is_alive() and not self.closed:
            raise RuntimeError(f"Session writer stopped; see {self.log_path}")
        if self.process and self.process.poll() is not None and not self.closed:
            raise RuntimeError(
                f"LeRobot writer exited ({self.process.returncode}); see {self.log_path}"
            )

    def begin(self):
        with self.lock:
            self.q.put_nowait({"command": "begin", "episode": self.saved})
            self.active = True

    def end(self):
        with self.lock:
            self.active = False
            self.q.put({"command": "end"}, timeout=5)

    def wait_reply(self, desired, tick, timeout=180):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            self.check()
            tick()
            try:
                reply = self.replies.get(timeout=0.05)
            except queue.Empty:
                continue
            if reply.get("error"):
                raise RuntimeError(reply["error"])
            if reply.get("event") == desired:
                return reply
        raise RuntimeError(f"Timed out waiting for writer {desired}")

    def run(self, latest, lock, viewer, stop):
        panel = viewer.server.gui.add_markdown("Session starting") if viewer else None
        state = "WARMUP"
        from .capture_quality import CaptureQuality
        from .markers import Markers

        quality = CaptureQuality(
            Markers(self.meta["config"]["markers"], self.meta["rgb_intrinsics"])
        )
        health = {"ready": False, "reasons": ["Waiting for sensors"]}

        def tick():
            nonlocal health
            if stop.is_set():
                raise RuntimeError("Sensor acquisition stopped")
            with lock:
                current = latest.copy()
            health = quality.update(current)
            if viewer:
                viewer.update(current)
            if panel:
                panel.content = (
                    f"**{state}** · saved {self.saved}/{self.episodes}\n\n"
                    + (
                        f"Width {health['width_m'] * 1000:.1f} mm; capture ready"
                        if health["ready"]
                        else "**NOT READY:** " + "; ".join(health["reasons"])
                    )
                )
            self.check()
            return current

        self.wait_reply("ready", tick)
        print(
            f"\nSESSION · {self.episodes} episodes × {self.seconds:g}s · {self.root}\n"
            "Devices remain running between episodes. Warmup/reset frames are not saved.\n"
            "Move gently until tracking is 3/3. Keep the browser preview open.",
            flush=True,
        )
        start = time.monotonic()
        last_print = -1
        stable_since = None
        while time.monotonic() - start < self.warmup:
            current = tick()
            elapsed = int(time.monotonic() - start)
            if elapsed != last_print:
                print(
                    f"\rWARMUP {elapsed:2d}/{self.warmup:g}s · tracking "
                    f"{current.get('pose', {}).get('tracker_confidence', 0)}/3",
                    end="",
                    flush=True,
                )
                last_print = elapsed
            time.sleep(0.05)
        print(
            "\nWarmup complete. Starting requires valid jaw width and fresh confidence 3/3 for 1 second."
        )
        while self.saved < self.episodes:
            state = "READY — Enter to record; q to exit"
            print(
                f"\nEpisode {self.saved + 1}/{self.episodes} · Enter = record {self.seconds:g}s "
                "· q = finish (Ctrl+C also exits here)",
                flush=True,
            )
            while True:
                current = tick()
                now = time.monotonic()
                pose = current.get("pose", {})
                fresh = now - pose.get("arrival_s", -1e20) < 0.25
                good = fresh and health["ready"]
                stable_since = (stable_since or now) if good else None
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    line = sys.stdin.readline()
                    if not line or line.strip().lower() == "q":
                        return
                    if line.strip():
                        print("Press Enter to record, or type q to finish.", flush=True)
                        continue
                    if stable_since is None or now - stable_since < 1:
                        print(
                            "NOT STARTED: "
                            + (
                                "; ".join(health["reasons"])
                                or "Hold valid tracking and width for 1 second"
                            ),
                            flush=True,
                        )
                        continue
                    # Reserve enough for lossless compressed depth, RGB staging and encoded output.
                    required = max(512 * 1024**2, int(self.seconds * 45 * 1024**2))
                    if shutil.disk_usage(self.root).free < required:
                        print(
                            f"NOT STARTED: need about {required / 1024**3:.1f} GiB free for this episode.",
                            flush=True,
                        )
                        continue
                    break
            self.begin()
            state = "RECORDING — Ctrl+C ends this episode"
            print(
                f"RECORDING {self.saved + 1}/{self.episodes} · Ctrl+C = save early",
                flush=True,
            )
            start = time.monotonic()
            last_print = -1
            try:
                while time.monotonic() - start < self.seconds:
                    current = tick()
                    elapsed = int(time.monotonic() - start)
                    if elapsed != last_print:
                        confidence = current.get("pose", {}).get(
                            "tracker_confidence", 0
                        )
                        print(
                            f"\rREC {elapsed:3d}/{self.seconds:g}s · confidence {confidence}/3"
                            + (
                                " — LOW TRACKING"
                                if confidence < 3
                                else (
                                    " — WIDTH INVALID"
                                    if not health["ready"]
                                    else " — width OK       "
                                )
                            ),
                            end="",
                            flush=True,
                        )
                        last_print = elapsed
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("\nEnding this episode early.", flush=True)
            finally:
                self.end()
            state = "SAVING — keep devices running"
            print("\nSaving LeRobot v3 episode…", flush=True)
            reply = self.wait_reply("saved", tick)
            if reply["frames"]:
                self.saved += 1
                print(
                    f"SAVED episode {self.saved}/{self.episodes} · {reply['frames']} frames "
                    f"· low-confidence frames: {reply['low_confidence_frames']} "
                    f"· invalid-width frames: {reply.get('invalid_width_frames', 0)}",
                    flush=True,
                )
            else:
                print("No RGB frames received; empty episode discarded.", flush=True)
            stable_since = None
        print("\nRequested episodes collected.", flush=True)

    def close(self, error=None):
        if self.closed or self.process is None:
            return
        self.closed = True
        with self.lock:
            self.active = False
        try:
            if self.thread and self.thread.is_alive() and not self.failure:
                self.q.put(
                    {"command": "finish", "error": str(error) if error else None},
                    timeout=5,
                )
                self.q.put(None, timeout=5)
                self.thread.join(timeout=120)
                if self.thread.is_alive():
                    raise RuntimeError("Session writer shutdown timed out")
                self.process.stdin.close()
                code = self.process.wait(timeout=120)
                if code:
                    raise RuntimeError(f"LeRobot writer failed ({code})")
            else:
                raise RuntimeError(f"Writer failed: {self.failure}")
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=10)
            raise
