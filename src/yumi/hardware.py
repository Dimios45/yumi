"""Serial-selected RealSense acquisition; never select the first connected camera."""

import contextlib
import importlib.metadata
import json
import os
import queue
import select
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def sdk():
    import pyrealsense2 as rs

    return rs


def devices():
    rs = sdk()
    ctx = rs.context()
    time.sleep(1)
    result = []
    for product in (rs.product_line.D400, rs.product_line.T200):
        found = ctx.query_devices(int(product))
        for index in range(len(found)):
            try:
                d = found[index]
                result.append(
                    {
                        key: d.get_info(getattr(rs.camera_info, key))
                        for key in ("name", "serial_number", "firmware_version")
                    }
                )
            except RuntimeError as e:
                result.append(
                    {
                        "product_line": str(product),
                        "error": str(e),
                        "hint": "Install RealSense udev rules, reload and reconnect camera",
                    }
                )
    return result


def intrinsics(profile):
    i = profile.as_video_stream_profile().get_intrinsics()
    return {
        k: getattr(i, k)
        for k in ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs")
    } | {"model": str(i.model)}


def snapshot(serial, output):
    rs = sdk()
    p = rs.pipeline()
    c = rs.config()
    c.enable_device(serial)
    c.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    p.start(c)
    try:
        for _ in range(30):
            f = p.wait_for_frames(5000).get_color_frame()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(output), cv2.cvtColor(np.asanyarray(f.get_data()), cv2.COLOR_RGB2BGR)
        ):
            raise OSError(output)
        output.with_suffix(".json").write_text(
            json.dumps(intrinsics(f.profile), indent=2)
        )
    finally:
        p.stop()


def _write_samples(root, q):
    with (root / "samples.jsonl").open("w") as file:
        while True:
            item = q.get()
            try:
                if item is None:
                    break
                if item["kind"] in ("image", "world"):
                    rgb = item.pop("rgb")
                    if not cv2.imwrite(
                        str(root / item["rgb_path"]),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1],
                    ):
                        raise OSError("PNG write failed")
                elif item["kind"] == "depth":
                    np.save(
                        root / item["depth_path"],
                        item.pop("depth"),
                        allow_pickle=False,
                    )
                file.write(json.dumps(item, allow_nan=False) + "\n")
            finally:
                q.task_done()
        file.flush()
        os.fsync(file.fileno())


class Episode:
    """One raw episode directory, fed by sensor callbacks that may outlive it.

    COMPLETE is written only after the writer drained and synced every sample.
    """

    def __init__(self, root, meta):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "rgb").mkdir()
        (self.root / "depth").mkdir()
        if "world_camera" in meta:
            (self.root / "world").mkdir()
        (self.root / "metadata.json").write_text(
            json.dumps(meta | {"start_unix_s": time.time()}, indent=2)
        )
        self.started = time.monotonic()
        self.poses = self.low_poses = self.drops = 0
        self.first_drop_s = self.last_confidence = None
        self.q = queue.Queue(maxsize=256)
        self.errors = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            _write_samples(self.root, self.q)
        except Exception as e:  # noqa: BLE001 — surfaced by put() and close()
            self.errors.put(e)

    def put(self, item):
        if not self.thread.is_alive():
            raise RuntimeError("Episode writer stopped; capture aborted")
        try:
            self.q.put(item, timeout=0.5)
        except queue.Full as e:
            raise RuntimeError("Disk writer cannot keep up; capture aborted") from e

    def close(self, error=None):
        """Finish the episode; returns the error that marked it FAILED, else None."""
        if self.thread.is_alive():
            try:
                self.q.put(None, timeout=5)
            except queue.Full:
                error = error or RuntimeError("Writer stuck")
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                error = error or RuntimeError("Writer shutdown timed out")
        if error is None and not self.errors.empty():
            error = self.errors.get()
        if error is not None:
            (self.root / "FAILED.txt").write_text(str(error))
            return error
        (self.root / "COMPLETE").write_text(
            "Raw capture closed successfully; not yet quality validated.\n"
        )
        return None


@contextlib.contextmanager
def _keyboard(requests):
    """Space toggles an episode, q quits. Without a TTY, each stdin line acts the same."""
    done = threading.Event()
    fd = sys.stdin.fileno()
    tty_mode = sys.stdin.isatty()
    if tty_mode:
        import termios
        import tty

        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    def read():
        while not done.is_set():
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            key = os.read(fd, 1) if tty_mode else sys.stdin.readline().encode()
            if not key:
                return
            key = key.strip(b"\n") or b" "
            if key[:1] == b" ":
                requests.put("toggle")
            elif key[:1] in (b"q", b"Q"):
                requests.put("quit")

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        yield
    finally:
        done.set()
        reader.join(timeout=1)
        if tty_mode:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


_CONFIDENCE = {0: ("FAILED", "41"), 1: ("LOW", "41"), 2: ("MEDIUM", "43"), 3: ("HIGH", "42")}


def _confidence_badge(confidence, tracking=True):
    """T265 confidence as a coloured terminal badge (green HIGH, yellow MEDIUM, red LOW)."""
    if not tracking:
        return "no tracker"
    if confidence is None:
        return "T265 starting…"
    name, colour = _CONFIDENCE.get(confidence, ("?", "41"))
    badge = f" T265 {confidence}/3 {name} "
    return f"\033[1;30;{colour}m{badge}\033[0m" if sys.stdout.isatty() else badge


def _next_episode(session):
    taken = [
        int(p.name.split("-")[1])
        for p in session.glob("episode-*")
        if p.name.split("-")[1].isdigit()
    ]
    return session / f"episode-{max(taken, default=-1) + 1:03d}"


def record(
    config,
    output,
    seconds=None,
    visualize=True,
    port=8080,
    tracking=True,
    session=None,
    interactive=False,
    world_serial=None,
):
    """Lossless raw episodes, independent sensor polling threads and bounded writer queues.

    Non-interactive: one episode at ``output`` lasting ``seconds`` (default 30).
    Interactive: sensors start once and stay running so the T265 can warm up;
    space starts and stops ``output/episode-NNN`` (auto-stopping after
    ``seconds`` if given), q quits. Samples outside an episode are not saved.

    The live T265 confidence is always shown; each episode's confidence
    statistics are saved to ``tracking.json``.

    ``session`` (record-session) receives every sample and owns the episode
    loop and storage; no raw episode directory is written.

    ``world_serial`` adds a fixed external (world) RGB camera, saved as
    ``kind: "world"`` samples under ``world/``.

    Queue overload aborts capture; it never silently drops a frame.
    """
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive")
    if not interactive and seconds is None:
        seconds = 30
    rs = sdk()
    root = Path(output)
    if session is None and not interactive and root.exists():
        raise FileExistsError(f"{root} already exists")
    stop = threading.Event()
    errors = queue.Queue()
    latest = {}
    lock = threading.Lock()
    active = {"episode": None}
    finished = []
    requests = queue.Queue()
    pipelines = []
    workers = []
    meta = {
        "config": config,
        "sdk_version": importlib.metadata.version("pyrealsense2"),
        "tracking": tracking,
    }

    def put(item):
        if session is not None:
            session.accept(item)
            return
        with lock:
            episode = active["episode"]
            if episode is not None and item["kind"] == "pose":
                episode.poses += 1
                if item["tracker_confidence"] < 3:
                    if episode.last_confidence is None or episode.last_confidence >= 3:
                        episode.drops += 1
                    episode.low_poses += 1
                    if episode.first_drop_s is None:
                        episode.first_drop_s = item["arrival_s"] - episode.started
                episode.last_confidence = item["tracker_confidence"]
        if episode is not None:
            episode.put(item)

    def start_episode():
        with lock:
            if active["episode"] is not None:
                return
        path = _next_episode(root) if interactive else root
        confidence = latest.get("pose", {}).get("tracker_confidence")
        episode = Episode(path, meta)
        with lock:
            active["episode"] = episode
        if interactive:
            warning = (
                ""
                if not tracking or confidence == 3
                else f"  WARNING: T265 confidence is {confidence}, not 3"
            )
            print(f"\n● REC  {path}{warning}", flush=True)

    def stop_episode(error=None):
        with lock:
            episode, active["episode"] = active["episode"], None
        if episode is None:
            return
        solid = 1 - episode.low_poses / max(episode.poses, 1)
        if tracking:
            (episode.root / "tracking.json").write_text(
                json.dumps(
                    {
                        "pose_samples": episode.poses,
                        "confidence_3_fraction": solid,
                        "confidence_drops": episode.drops,
                        "first_drop_s": episode.first_drop_s,
                    },
                    indent=2,
                )
            )
        failure = episode.close(error)
        finished.append(episode.root)
        if failure is not None:
            errors.put(failure)
        elif interactive:
            verdict = (
                ""
                if not tracking
                else " · confidence 3 throughout ✓"
                if episode.low_poses == 0
                else f" · confidence 3 for {solid * 100:.0f}% ({episode.drops} drop(s))"
            )
            print(
                f"\n■ saved {episode.root} "
                f"({time.monotonic() - episode.started:.1f} s){verdict}",
                flush=True,
            )

    def guarded(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — propagate worker/shutdown failures to caller
            errors.put(e)
            stop.set()

    def stamp(frame):
        return {
            "device_s": frame.get_timestamp() / 1000,
            "arrival_s": time.monotonic(),
            "frame_number": frame.get_frame_number(),
            "domain": str(frame.get_frame_timestamp_domain()),
        }

    def video_callback(frame):
        if stop.is_set():
            return

        def accept():
            item = stamp(frame)
            index = frame.get_frame_number()
            if frame.profile.stream_type() == rs.stream.color:
                rgb = np.asanyarray(frame.get_data()).copy()
                item |= {
                    "kind": "image",
                    "index": index,
                    "rgb_path": f"rgb/{index:08d}.png",
                    "rgb": rgb,
                }
                with lock:
                    latest["rgb"] = rgb
                    latest["rgb_arrival_s"] = item["arrival_s"]
            elif frame.profile.stream_type() == rs.stream.depth:
                item |= {
                    "kind": "depth",
                    "depth_path": f"depth/{index:08d}.npy",
                    "depth": np.asanyarray(frame.get_data()).copy(),
                }
            else:
                return
            put(item)

        guarded(accept)

    def pose_loop(p):
        last_frame = {}
        while not stop.is_set():
            frames = p.wait_for_frames(2000)
            for f in frames:
                # A frameset may repeat the last pose/IMU sample when another
                # stream advances. Persist each sensor sample exactly once.
                stream = str(f.profile.stream_type())
                number = f.get_frame_number()
                previous = last_frame.get(stream, -1)
                if number == previous:
                    continue
                if number < previous:
                    raise RuntimeError("T265 frame counter reset; restart the episode")
                last_frame[stream] = number
                item = stamp(f)
                if f.is_pose_frame():
                    d = f.as_pose_frame().get_pose_data()
                    item |= {
                        "kind": "pose",
                        "position": [d.translation.x, d.translation.y, d.translation.z],
                        "quaternion_xyzw": [
                            d.rotation.x,
                            d.rotation.y,
                            d.rotation.z,
                            d.rotation.w,
                        ],
                        "tracker_confidence": d.tracker_confidence,
                        "mapper_confidence": d.mapper_confidence,
                        "velocity": [d.velocity.x, d.velocity.y, d.velocity.z],
                        "angular_velocity": [
                            d.angular_velocity.x,
                            d.angular_velocity.y,
                            d.angular_velocity.z,
                        ],
                    }
                    with lock:
                        latest["pose"] = item.copy()

                elif f.is_motion_frame():
                    d = f.as_motion_frame().get_motion_data()
                    item |= {
                        "kind": str(f.profile.stream_type()).split(".")[-1],
                        "xyz": [d.x, d.y, d.z],
                    }
                else:
                    continue
                put(item)

    def world_loop(p):
        last = -1
        while not stop.is_set():
            f = p.wait_for_frames(2000).get_color_frame()
            if not f or f.get_frame_number() == last:
                continue
            last = f.get_frame_number()
            rgb = np.asanyarray(f.get_data()).copy()
            item = stamp(f) | {
                "kind": "world",
                "index": last,
                "rgb_path": f"world/{last:08d}.png",
                "rgb": rgb,
            }
            with lock:
                latest["world"] = rgb
            put(item)

    viewer = None
    camera_sensor = None
    camera_started = False
    keyboard = contextlib.nullcontext()
    try:
        ctx = rs.context()
        candidates = [
            d
            for d in ctx.query_devices(int(rs.product_line.D400))
            if d.get_info(rs.camera_info.serial_number) == config["rgb_serial"]
        ]
        if len(candidates) != 1 or "405" not in candidates[0].get_info(
            rs.camera_info.name
        ):
            raise ValueError("Recording RGB serial must identify a connected D405")
        device = candidates[0]
        w, h, fps = config["resolution"][0], config["resolution"][1], config["fps"]
        selected = None
        for sensor in device.query_sensors():
            profiles = {}
            for profile in sensor.get_stream_profiles():
                if not profile.is_video_stream_profile():
                    continue
                video = profile.as_video_stream_profile()
                if video.width() == w and video.height() == h and profile.fps() == fps:
                    profiles[(profile.stream_type(), profile.format())] = profile
            if (rs.stream.color, rs.format.rgb8) in profiles and (
                rs.stream.depth,
                rs.format.z16,
            ) in profiles:
                selected = (
                    sensor,
                    profiles[(rs.stream.color, rs.format.rgb8)],
                    profiles[(rs.stream.depth, rs.format.z16)],
                )
                break
        if selected is None:
            raise ValueError(
                "D405 does not expose the requested RGB/depth profiles on one sensor"
            )
        sensor, color_profile, depth_profile = selected
        sensor.open([color_profile, depth_profile])
        camera_sensor = sensor
        ex = depth_profile.get_extrinsics_to(color_profile)
        meta |= {
            "acquisition": "independent native sensor callbacks; no frameset synchronization",
            "rgb_intrinsics": intrinsics(color_profile),
            "depth_intrinsics": intrinsics(depth_profile),
            "depth_scale_m": device.first_depth_sensor().get_depth_scale(),
            "depth_to_color": {
                "rotation_column_major": ex.rotation,
                "translation_m": ex.translation,
            },
            "rgb_firmware": device.get_info(rs.camera_info.firmware_version),
        }
        worker_fns = []
        if tracking:
            if not config.get("tracking_serial"):
                raise ValueError("Set T265 serial explicitly")
            tp = rs.pipeline()
            tc = rs.config()
            tc.enable_device(config["tracking_serial"])
            tc.enable_stream(rs.stream.pose)
            tc.enable_stream(rs.stream.accel)
            tc.enable_stream(rs.stream.gyro)
            tprofile = tp.start(tc)
            pipelines.append(tp)
            meta["tracking_firmware"] = tprofile.get_device().get_info(
                rs.camera_info.firmware_version
            )
            worker_fns.append(lambda: pose_loop(tp))
        if world_serial:
            wp = rs.pipeline()
            wc = rs.config()
            wc.enable_device(world_serial)
            wc.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
            wprofile = wp.start(wc)
            pipelines.append(wp)
            meta["world_camera"] = {
                "serial": world_serial,
                "name": wprofile.get_device().get_info(rs.camera_info.name),
                "intrinsics": intrinsics(wprofile.get_stream(rs.stream.color)),
                "firmware": wprofile.get_device().get_info(
                    rs.camera_info.firmware_version
                ),
            }
            worker_fns.append(lambda: world_loop(wp))
        if session is not None:
            session.setup(meta)
        elif interactive:
            root.mkdir(parents=True, exist_ok=True)
            meta["session_started_unix_s"] = time.time()
        else:
            start_episode()
        camera_sensor.start(video_callback)
        camera_started = True
        for fn in worker_fns:
            worker = threading.Thread(target=lambda f=fn: guarded(f), daemon=True)
            worker.start()
            workers.append(worker)
        if visualize:
            from .viewer import Viewer

            viewer = Viewer(
                config,
                meta["rgb_intrinsics"],
                port,
                on_toggle=(lambda: requests.put("toggle")) if interactive else None,
            )
        if interactive:
            keyboard = _keyboard(requests)
            keyboard.__enter__()
            print(
                "Warm up the T265 (move slowly until confidence 3).\n"
                "SPACE start/stop episode · q quit"
                + (f" · episodes auto-stop after {seconds:g} s" if seconds else ""),
                flush=True,
            )
        session_start, last_status = time.monotonic(), 0.0
        if session is not None:
            session.run(latest, lock, viewer, stop)
        while session is None and not stop.wait(0.05):
            with lock:
                current = latest.copy()
                episode = active["episode"]
            elapsed = time.monotonic() - (episode.started if episode else session_start)
            if not interactive and elapsed >= seconds:
                break
            quit_requested = False
            while not requests.empty():
                request = requests.get()
                if request == "quit":
                    quit_requested = True
                elif episode is None:
                    start_episode()
                else:
                    stop_episode()
            if quit_requested:
                break
            with lock:
                episode = active["episode"]
            elapsed = time.monotonic() - (episode.started if episode else session_start)
            if interactive and episode and seconds and elapsed >= seconds:
                stop_episode()
                episode = None
            confidence = current.get("pose", {}).get("tracker_confidence")
            state = (
                f"● REC {episode.root.name} {elapsed:5.1f} s"
                if episode
                else f"idle · {len(finished)} saved · SPACE to record"
            )
            if viewer:
                viewer.update(current, state if interactive else None)
            if interactive and time.monotonic() - last_status > 0.2:
                last_status = time.monotonic()
                print(f"\r  {_confidence_badge(confidence, tracking)} · {state}    ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001 — propagate worker/shutdown failures to caller
        errors.put(e)
    finally:
        stop.set()
        keyboard.__exit__(None, None, None)
        if camera_sensor is not None:
            try:
                if camera_started:
                    camera_sensor.stop()
                camera_sensor.close()
            except RuntimeError as e:
                errors.put(e)
        for worker in workers:
            worker.join(timeout=5)
        if any(t.is_alive() for t in workers):
            errors.put(RuntimeError("Sensor worker did not stop"))
        for p in reversed(pipelines):
            try:
                p.stop()
            except Exception as e:  # noqa: BLE001 — propagate worker/shutdown failures to caller
                errors.put(e)
        stop_episode(None if errors.empty() else errors.queue[0])
        if session is not None:
            try:
                session.close(
                    None
                    if errors.empty()
                    else RuntimeError("Capture failed; see FAILED.txt")
                )
            except Exception as e:  # noqa: BLE001 — propagate shutdown failures
                errors.put(e)
        if viewer:
            viewer.close()
        if interactive:
            print(flush=True)
    if not errors.empty():
        raise errors.get()
    return finished if interactive else root
