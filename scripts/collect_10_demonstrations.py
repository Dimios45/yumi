#!/usr/bin/env python3
"""Collect ten accepted handheld demonstrations and export paired LeRobot v3 datasets.

Run with the legacy RealSense environment. No robot or motor connection is made.
"""

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from yumi.capture_quality import CaptureQuality
from yumi.demonstrations import (
    complete_episode_segment,
    prepare_episode,
    write_manifest,
)
from yumi.hardware import Episode, _keyboard, record
from yumi.markers import Markers

ROOT = Path(__file__).resolve().parents[1]


class Collection:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.lock = threading.Lock()
        self.active = None
        self.manifests = []
        self.errors = []
        self.keys = queue.Queue()  # "toggle" (space) / "quit" (q)

    def key(self):
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None

    def setup(self, meta):
        self.meta = meta
        self.quality = CaptureQuality(
            Markers(self.config["markers"], meta["rgb_intrinsics"])
        )

    def accept(self, item):
        with self.lock:
            if self.active is not None:
                self.active.put(item)

    def close(self, error=None):
        with self.lock:
            episode, self.active = self.active, None
        if episode is not None:
            episode.close(error or RuntimeError("Collection interrupted"))
        self.save_report()

    def save_report(self):
        path = self.args.output / "collection-report.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {"accepted": [str(p) for p in self.manifests], "rejected": self.errors},
                indent=2,
            )
        )
        temp.replace(path)

    def tick(self, latest, lock, viewer):
        with lock:
            current = latest.copy()
        health = self.quality.update(current)
        if viewer:
            viewer.update(current)
        return health

    def wait_start(self, latest, lock, viewer, stop):
        stable_since = None
        printed = None
        while not stop.is_set():
            health = self.tick(latest, lock, viewer)
            now = time.monotonic()
            stable_since = (stable_since or now) if health["ready"] else None
            ready = stable_since is not None and now - stable_since >= 1
            status = (
                "READY — SPACE to start recording, q to quit"
                if ready
                else "NOT READY: "
                + (
                    "; ".join(health["reasons"])
                    or "hold valid tracking/width for one second"
                )
            )
            if status != printed:
                print(status, flush=True)
                printed = status
            key = self.key()
            if key == "quit":
                return False
            if key == "toggle" and ready:
                return True
            stop.wait(0.05)
        return False

    def run(self, latest, lock, viewer, stop):
        with _keyboard(self.keys):
            self._run(latest, lock, viewer, stop)

    def _run(self, latest, lock, viewer, stop):
        a = self.args
        print(
            f"Warming up for {a.warmup:g}s. Move gently to establish T265 tracking; keep both tags visible.",
            flush=True,
        )
        until = time.monotonic() + a.warmup
        while time.monotonic() < until and not stop.wait(0.05):
            self.tick(latest, lock, viewer)
        for attempt in range(a.max_attempts):
            if len(self.manifests) == a.count or stop.is_set():
                break
            print(
                f"Accepted {len(self.manifests)}/{a.count}. Return handheld to starting pose."
                + (f" Recordings auto-stop after {a.seconds:g}s." if a.seconds else ""),
                flush=True,
            )
            if not self.wait_start(latest, lock, viewer, stop):
                break
            raw = a.output / "raw" / f"episode-{attempt:03d}"
            episode = Episode(raw, self.meta)
            with self.lock:
                self.active = episode
            while self.key() is not None:  # drop keys pressed before the start
                pass
            started = time.monotonic()
            print("RECORDING — SPACE to stop.", flush=True)
            last_status, quit_requested = -1, False
            try:
                while not stop.wait(0.05):
                    if a.seconds and time.monotonic() - started >= a.seconds:
                        print(f"Auto-stopped at {a.seconds:g}s.", flush=True)
                        break
                    key = self.key()
                    if key is not None:
                        quit_requested = key == "quit"
                        break
                    health = self.tick(latest, lock, viewer)
                    elapsed = int(time.monotonic() - started)
                    if elapsed != last_status:
                        print(
                            f"REC {elapsed}s: "
                            + (
                                f"width {health['width_m'] * 1000:.1f} mm; tracking OK"
                                if health["ready"]
                                else "; ".join(health["reasons"])
                            ),
                            flush=True,
                        )
                        last_status = elapsed
            finally:
                with self.lock:
                    self.active = None
                error = episode.close(
                    RuntimeError("Sensor capture stopped unexpectedly")
                    if stop.is_set()
                    else None
                )
            if error:
                raise error
            try:
                doc = prepare_episode(
                    raw,
                    self.config,
                    allow_provisional=a.allow_provisional,
                    min_seconds=a.min_seconds,
                    require_depth=True,
                    with_world=bool(a.world_serial),
                )
                write_manifest(doc, raw / "quality-report.json")
                # One physical attempt is one demonstration; do not splice a
                # discontinuous capture into multiple nominal demonstrations.
                max_trim = int(a.max_edge_trim_seconds * self.config["fps"])
                complete_episode_segment(doc, max_trim)
                capture = raw / "capture-manifest.json"
                labelled = raw / "joint-manifest.json"
                write_manifest(doc, capture)
                command = [
                    str(a.ik_python),
                    "-m",
                    "yumi.joint_labels",
                    "--manifest",
                    str(capture),
                    "--profile",
                    str(a.profile),
                    "--karma-root",
                    str(a.karma_root),
                    "--arm-xml",
                    str(a.arm_xml),
                    "--output",
                    str(labelled),
                ]
                if a.allow_provisional:
                    command.append("--allow-provisional")
                subprocess.run(
                    command,
                    check=True,
                    env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
                )
                self.manifests.append(labelled)
                print(
                    "Accepted; full-pose IK, limits, rates and model collisions passed."
                )
            except (ValueError, subprocess.CalledProcessError) as e:
                self.errors.append({"raw": str(raw), "reason": str(e)})
                (raw / "REJECTED.json").write_text(
                    json.dumps(self.errors[-1], indent=2)
                )
                print(f"Rejected, raw data retained: {e}")
            self.save_report()
            if quit_requested:
                break
        self.save_report()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--profile", type=Path, default=ROOT / "configs/yam.collection-provisional.json"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=10)
    p.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Auto-stop a recording after this long (0 = only SPACE stops it)",
    )
    p.add_argument("--min-seconds", type=float, default=4)
    p.add_argument("--max-attempts", type=int, default=30)
    p.add_argument(
        "--task", required=True, help="Actual task demonstrated in every episode"
    )
    p.add_argument("--warmup", type=float, default=20)
    p.add_argument("--max-edge-trim-seconds", type=float, default=0.2)
    p.add_argument("--world-serial", help="Optional fixed RGB camera serial")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--karma-root", type=Path, default=Path("/home/yambox/karma"))
    p.add_argument(
        "--arm-xml",
        type=Path,
        default=Path("/home/yambox/karma/i2rt/i2rt/robot_models/arm/yam/yam.xml"),
    )
    p.add_argument(
        "--ik-python", type=Path, default=Path("/home/yambox/karma/.venv/bin/python")
    )
    p.add_argument(
        "--export-python", type=Path, default=ROOT / "exporter/.venv/bin/python"
    )
    p.add_argument("--allow-provisional", action="store_true")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    if (
        a.count < 1
        or a.max_attempts < a.count
        or a.min_seconds <= 0
        or (a.seconds and a.seconds <= a.min_seconds)
    ):
        p.error(
            "Require positive count, enough attempts, and 0 < min-seconds < seconds"
        )
    c = json.loads(a.config.read_text())
    c["task"] = a.task
    if a.warmup < 0 or a.max_edge_trim_seconds < 0:
        p.error("Warmup and permitted edge trimming must be nonnegative")
    profile = json.loads(a.profile.read_text())
    from yumi.geometry import transform

    transform(c.get("T_tracker_tcp"))
    transform(profile["T_capture_tcp_robot_tool"])
    if not a.allow_provisional and not all(
        (
            c.get("calibration_verified"),
            c.get("timing_verified"),
            profile.get("registration_verified"),
        )
    ):
        p.error("Unverified calibration requires explicit --allow-provisional")
    for path in (a.ik_python, a.export_python, a.arm_xml):
        if not path.is_file():
            p.error(f"Missing dependency: {path}")
    subprocess.run([str(a.ik_python), "-c", "import mujoco, scipy"], check=True)
    subprocess.run(
        [
            str(a.export_python),
            "-c",
            "from lerobot.datasets.lerobot_dataset import LeRobotDataset; import av",
        ],
        check=True,
    )
    print(
        f"Home reference: {profile['initial_joints_rad']}; labels are offline IK, gripper in metres."
    )
    if a.preflight:
        return 0
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "capture-config.json").write_text(json.dumps(c, indent=2))
    (a.output / "robot-profile.json").write_text(json.dumps(profile, indent=2))
    a.profile = a.output / "robot-profile.json"
    session = Collection(a, c)
    record(
        c,
        a.output,
        session=session,
        visualize=not a.no_viewer,
        port=a.port,
        world_serial=a.world_serial,
    )
    if len(session.manifests) != a.count:
        print(
            f"Collected {len(session.manifests)}/{a.count}; raw data/report retained. No complete paired export produced."
        )
        return 2
    command = [
        str(a.export_python),
        str(ROOT / "exporter/export_demonstrations.py"),
        "--manifests",
        *map(str, session.manifests),
        "--output",
        str(a.output / "datasets"),
        "--expected-episodes",
        str(a.count),
    ]
    if a.allow_provisional:
        command.append("--allow-provisional")
    if a.world_serial:
        command.append("--with-world")
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
