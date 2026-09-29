"""Official LeRobot v3 raw-observation writer; private trusted local pipe protocol.

No action targets are fabricated. Provisional state uses the most recently
received pose, without claiming time alignment; full sensor logs support later
reprocessing. Validity masks distinguish missing data from real zero values.
"""

import argparse
import json
import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np


def receive(pipe):
    header = pipe.read(8)
    if not header:
        raise EOFError("Capture connection closed without finish")
    if len(header) != 8:
        raise EOFError("Truncated packet header")
    size = struct.unpack("!Q", header)[0]
    if size > 64 * 1024**2:
        raise ValueError("Oversized capture packet")
    data = pipe.read(size)
    if len(data) != size:
        raise EOFError("Truncated capture packet")
    return pickle.loads(data)


def run(root, repo_id, task, source, replies):
    # Third-party progress output must not mix with the reply protocol.
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"Expected LeRobot v3, got {CODEBASE_VERSION}")

    def reply(**value):
        replies.write(json.dumps(value) + "\n")
        replies.flush()

    setup = receive(source)
    meta = setup["metadata"]
    w, h = meta["config"]["resolution"]
    features = {
        "observation.images.d405": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "aperture_m"],
        },
        "observation.quality": {
            "dtype": "float32",
            "shape": (4,),
            "names": [
                "tracker_confidence",
                "pose_valid",
                "width_valid",
                "image_minus_pose_s",
            ],
        },
        "observation.sensor_time": {
            "dtype": "float64",
            "shape": (2,),
            "names": ["device_s", "arrival_monotonic_s"],
        },
    }
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=meta["config"]["fps"],
        robot_type="custom_umi_t265_d405_raw",
        features=features,
        image_writer_threads=2,
        video_backend="pyav",
        vcodec="h264",
    )
    (root / "session.json").write_text(
        json.dumps(
            {
                **meta,
                "schema": "LeRobot v3 raw observations; not training-ready actions",
                "state_semantics": "Provisional TCP pose in persistent T265 session frame; last received pose, not time-aligned; aperture metres.",
                "invalid_values": "If pose_valid=0, position/orientation must not be used. Missing pose uses identity. If width_valid=0, aperture=0 is a placeholder, not a closed-jaw measurement.",
                "timestamps": "LeRobot nominal FPS timestamps index captured RGB frames; observation.sensor_time preserves actual exposure/arrival times. No missing frames are filled.",
                "action_semantics": "No action feature: calibration, synchronization, retargeting and training action construction remain separate.",
            },
            indent=2,
        )
        + "\n"
    )
    log = None
    count = low = invalid_width = 0
    extras = None
    failed = False

    def save(interrupted=False):
        nonlocal log, count, low, invalid_width
        if log is None:
            return
        log.flush()
        os.fsync(log.fileno())
        log.close()
        log = None
        if count:
            ds.save_episode()
            (extras / "capture.json").write_text(
                json.dumps(
                    {
                        "frames": count,
                        "low_confidence_frames": low,
                        "invalid_width_frames": invalid_width,
                        "interrupted": interrupted,
                        "quality_validated": False,
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            ds.clear_episode_buffer()
        reply(
            event="saved",
            frames=count,
            low_confidence_frames=low,
            invalid_width_frames=invalid_width,
        )
        count = low = invalid_width = 0

    try:
        reply(event="ready")
        while True:
            packet = receive(source)
            command = packet.get("command")
            if command == "begin":
                if log is not None:
                    raise RuntimeError("Episode already open")
                extras = root / "extras" / f"episode_{packet['episode']:06d}"
                extras.mkdir(parents=True, exist_ok=True)
                (extras / "depth").mkdir(exist_ok=True)
                (extras / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
                log = (extras / "samples.jsonl").open("w")
                count = low = invalid_width = 0
            elif command == "end":
                save()
            elif command == "finish":
                save(interrupted=True)
                failed = bool(packet.get("error"))
                if failed:
                    (root / "FAILED.txt").write_text(packet["error"])
                break
            elif "sample" in packet:
                if log is None:
                    raise RuntimeError("Sample outside episode")
                item = packet["sample"]
                if item["kind"] == "image":
                    rgb = item.pop("rgb")
                    state = item.pop("provisional_state")
                    quality = item.pop("quality")
                    item.pop("rgb_path", None)
                    item["lerobot_frame_index"] = count
                    ds.add_frame(
                        {
                            "observation.images.d405": rgb,
                            "observation.state": state,
                            "observation.quality": quality,
                            "observation.sensor_time": np.array(
                                [item["device_s"], item["arrival_s"]], dtype=np.float64
                            ),
                            "task": task,
                        }
                    )
                    count += 1
                    low += int(quality[0] < 3)
                    invalid_width += int(quality[2] == 0)
                elif item["kind"] == "depth":
                    depth = item.pop("depth")
                    path = f"depth/{item['frame_number']:08d}.npz"
                    np.savez_compressed(extras / path, depth=depth)
                    item["depth_path"] = path
                log.write(json.dumps(item, allow_nan=False) + "\n")
    except BaseException as e:
        failed = True
        (root / "FAILED.txt").write_text(str(e))
        reply(error=str(e))
        raise
    finally:
        if log is not None:
            log.close()
        ds.finalize()
        if ds.image_writer:
            ds.image_writer.stop()
    if not failed:
        (root / "COMPLETE").write_text(
            "Session closed; raw observations, not calibration/quality validation.\n"
        )
    reply(event="finished")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--task", required=True)
    a = p.parse_args()
    replies = sys.stdout
    sys.stdout = sys.stderr
    run(a.root, a.repo_id, a.task, sys.stdin.buffer, replies)
