"""Serial-selected RealSense acquisition; never select the first connected camera."""

import importlib.metadata
import json
import queue
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


def record(config, output, seconds, visualize=True, port=8080, tracking=True):
    """Lossless raw episode, independent sensor polling threads and bounded writer queue.

    A COMPLETE marker is created only after orderly shutdown and all writes.
    Queue overload aborts capture; it never silently drops a frame.
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    rs = sdk()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "rgb").mkdir()
    (root / "depth").mkdir()
    q = queue.Queue(maxsize=256)
    stop = threading.Event()
    errors = queue.Queue()
    latest = {}
    lock = threading.Lock()
    pipelines = []
    workers = []
    meta = {
        "config": config,
        "sdk_version": importlib.metadata.version("pyrealsense2"),
        "start_unix_s": time.time(),
        "tracking": tracking,
    }

    def put(item):
        try:
            q.put(item, timeout=0.5)
        except queue.Full as e:
            raise RuntimeError("Disk writer cannot keep up; capture aborted") from e

    def guarded(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — propagate worker/shutdown failures to caller
            errors.put(e)
            stop.set()

    def writer():
        with (root / "samples.jsonl").open("w") as file:
            while True:
                item = q.get()
                try:
                    if item is None:
                        break
                    if item["kind"] == "image":
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
            import os

            os.fsync(file.fileno())

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

    viewer = None
    writer_thread = None
    camera_sensor = None
    camera_started = False
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
        (root / "metadata.json").write_text(json.dumps(meta, indent=2))
        writer_thread = threading.Thread(target=lambda: guarded(writer), daemon=True)
        writer_thread.start()
        camera_sensor.start(video_callback)
        camera_started = True
        for fn in worker_fns:
            worker = threading.Thread(target=lambda f=fn: guarded(f), daemon=True)
            worker.start()
            workers.append(worker)
        if visualize:
            from .viewer import Viewer

            viewer = Viewer(config, meta["rgb_intrinsics"], port)
        end = time.monotonic() + seconds
        while time.monotonic() < end and not stop.wait(0.05):
            if viewer:
                with lock:
                    current = latest.copy()
                viewer.update(current)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001 — propagate worker/shutdown failures to caller
        errors.put(e)
    finally:
        stop.set()
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
        if writer_thread and writer_thread.is_alive():
            try:
                q.put(None, timeout=5)
            except queue.Full:
                errors.put(RuntimeError("Writer stuck"))
            writer_thread.join(timeout=30)
            if writer_thread.is_alive():
                errors.put(RuntimeError("Writer shutdown timed out"))
        if viewer:
            viewer.close()
    if not errors.empty():
        error = errors.get()
        (root / "FAILED.txt").write_text(str(error))
        raise error
    (root / "COMPLETE").write_text(
        "Raw capture closed successfully; not yet quality validated.\n"
    )
    return root
