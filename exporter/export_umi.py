"""Write/read back synchronized UMI chunks through the official LeRobot v3 API."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

CONTRACT = "yumi.umi.relative_tcp_10d.v1"


def export(manifests, root, repo_id, allow_provisional=False):
    if CODEBASE_VERSION != "v3.0":
        raise ValueError("LeRobot v3 required")
    root = Path(root)
    stage = root.with_name(root.name + ".partial")
    if root.exists() or stage.exists():
        raise FileExistsError("Use a new output directory")
    docs = [json.loads(Path(p).read_text()) for p in manifests]
    if not docs or not any(d["samples"] for d in docs):
        raise ValueError("No valid UMI chunks")
    if any(d["contract"] != CONTRACT for d in docs):
        raise ValueError("Contract mismatch")
    verified = all(d["calibration_verified"] for d in docs)
    if not verified and not allow_provisional:
        raise ValueError(
            "Calibration is provisional; explicit --allow-provisional required"
        )
    first = next(d for d in docs if d["samples"])
    history, horizon, fps = first["history"], first["horizon"], first["fps"]
    w, h = first["calibration"]["resolution"]
    image_keys = [
        f"observation.images.d405_history_{i}" for i in range(history - 1)
    ] + ["observation.images.d405"]
    features = {
        k: {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
        for k in image_keys
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": (history, 10),
                "names": None,
            },
            "action": {"dtype": "float32", "shape": (horizon, 10), "names": None},
            "observation.capture_time": {
                "dtype": "float64",
                "shape": (1,),
                "names": ["seconds"],
            },
            "action.time_offsets": {
                "dtype": "float32",
                "shape": (horizon,),
                "names": None,
            },
        }
    )
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        root=stage,
        fps=fps,
        features=features,
        robot_type="single_arm_umi_relative_tcp",
        video_backend="pyav",
        vcodec="h264",
        image_writer_threads=2,
    )
    mapping = []
    caps = {}
    cache = {}

    def image(doc, frame):
        path = str(Path(doc["source_root"]) / frame["video_path"])
        index = frame["video_frame_index"]
        key = (path, index)
        if key not in cache:
            cap = (
                caps.setdefault(path, cv2.VideoCapture(path))
                if path not in caps
                else caps[path]
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode {path}:{index}")
            cache[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if len(cache) > history * 3:
                cache.pop(next(iter(cache)))
        return cache[key]

    try:
        for doc in docs:
            if (
                doc["history"],
                doc["horizon"],
                doc["fps"],
                doc["calibration"]["resolution"],
            ) != (history, horizon, fps, [w, h]):
                raise ValueError("Inconsistent dataset contract/dimensions")
            by_index = {f["frame_index"]: f for f in doc["frames"]}
            previous = None
            active = False
            for sample in doc["samples"]:
                anchor = by_index[sample["anchor_frame"]]
                # Never concatenate across rejected intervals or source episodes.
                if previous is not None and anchor["frame_index"] != previous + 1:
                    ds.save_episode()
                    active = False
                if not active:
                    mapping.append(
                        {
                            "output_episode": len(mapping),
                            "source_episode": doc["source_episode"],
                            "source_root": doc["source_root"],
                            "start_frame": anchor["frame_index"],
                        }
                    )
                    active = True
                frame = {
                    k: image(doc, by_index[index])
                    for k, index in zip(image_keys, sample["history_frames"])
                }
                frame.update(
                    {
                        "observation.state": np.asarray(
                            sample["observation_state"], np.float32
                        ),
                        "action": np.asarray(sample["action"], np.float32),
                        "observation.capture_time": np.array(
                            [anchor["time_s"]], np.float64
                        ),
                        "action.time_offsets": np.asarray(
                            sample["action_time_offsets_s"], np.float32
                        ),
                        "task": doc["task"],
                    }
                )
                if not np.isfinite(frame["action"]).all():
                    raise ValueError("Nonfinite action")
                ds.add_frame(frame)
                previous = anchor["frame_index"]
            if active:
                ds.save_episode()
        (stage / "umi-contract.json").write_text(
            json.dumps(
                {
                    "contract": CONTRACT,
                    "history": history,
                    "horizon": horizon,
                    "fps": fps,
                    "pose_layout": "xyz metres; rotation columns 0 then 1; aperture metres",
                    "relative_to": "current observation TCP for ALL history and action poses",
                    "history_image_keys": image_keys,
                    "calibration_verified": verified,
                    "robot_feasibility_verified": False,
                    "source_mapping": mapping,
                    "provenance": [
                        {"sha256": d["source_sha256"], "calibration": d["calibration"]}
                        for d in docs
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        ds.finalize()
        if ds.image_writer:
            ds.image_writer.stop()
        for cap in caps.values():
            cap.release()
    check = LeRobotDataset(repo_id, root=stage, video_backend="pyav")
    for i in range(len(check)):
        row = check[i]
        assert row["action"].shape == (horizon, 10)
        assert row["observation.state"].shape == (history, 10)
        for k in image_keys:
            assert row[k].shape == (3, h, w)
    report = {
        "format": "v3.0",
        "frames": len(check),
        "episodes": check.num_episodes,
        "all_frames_read_back": True,
        "calibration_verified": verified,
    }
    (stage / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    stage.rename(root)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default="local/single-arm-umi")
    p.add_argument("--allow-provisional", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            export(a.manifests, a.root, a.repo_id, a.allow_provisional), indent=2
        )
    )
