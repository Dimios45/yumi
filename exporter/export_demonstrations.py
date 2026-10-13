"""Export matching EE and six-arm-joint + aperture LeRobot v3 datasets.

Run in the pinned exporter environment. Joint values come from verified offline
IK manifests, not from this writer. Raw depth/IMU and calibration provenance are
preserved in extras; native depth is never encoded as lossy RGB video.
"""

import argparse
import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

CONTRACT = "yumi.demonstration.ee_joint7.v1"
EE_NAMES = ["x_m", "y_m", "z_m", "r00", "r10", "r20", "r01", "r11", "r21", "aperture_m"]
JOINT_NAMES = [f"joint{i}_rad" for i in range(1, 7)] + ["aperture_m"]


def matrix(pose):
    p = np.asarray(pose, float)
    if p.shape != (10,) or not np.isfinite(p).all() or p[9] < 0:
        raise ValueError("Expected finite metric 10D pose")
    r = np.column_stack((p[3:6], p[6:9], np.cross(p[3:6], p[6:9])))
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5):
        raise ValueError("Invalid stored rotation columns")
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p[:3]
    return t


def encode(t, width):
    return np.r_[t[:3, 3], t[:3, 0], t[:3, 1], width].astype(np.float32)


def features(w, h, mode, with_world):
    video = {
        "dtype": "video",
        "shape": (h, w, 3),
        "names": ["height", "width", "channels"],
    }
    ee = {"dtype": "float32", "shape": (10,), "names": EE_NAMES}
    state = (
        ee
        if mode == "ee"
        else {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES}
    )
    f = {
        "observation.images.wrist": video,
        "observation.state": state,
        "action": state,
        "observation.ee_pose": ee,
        "observation.ee_pose.world": ee,
        "observation.ee_pose.robot": ee,
        "action.ee_relative": ee,
        "observation.relative_state": {
            "dtype": "float32",
            "shape": (11,),
            "names": None,
        },
        "observation.gripper_width_measured": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["measured"],
        },
        "observation.sensor_time": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["corrected_capture_s", "device_s", "arrival_monotonic_s"],
        },
        "action.time_offset": {"dtype": "float32", "shape": (1,), "names": ["seconds"]},
        "observation.ik_error": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["position_m", "rotation_rad"],
        },
    }
    if with_world:
        f["observation.images.world"] = video
    return f


def frame_values(frames, i, mode, obs_step):
    cur, nxt, prev = frames[i], frames[i + 1], frames[max(0, i - obs_step)]
    now = matrix(cur["ee_pose"])
    past = np.linalg.inv(now) @ matrix(prev["ee_pose"])
    delta = np.linalg.inv(now) @ matrix(nxt["ee_pose"])
    key = "ee_pose" if mode == "ee" else "joint_state"
    state, action = np.asarray(cur[key], np.float32), np.asarray(nxt[key], np.float32)
    if not np.isfinite([state, action]).all():
        raise ValueError("Nonfinite state/action")
    return {
        "observation.state": state,
        "action": action,
        "observation.ee_pose": np.asarray(cur["ee_pose"], np.float32),
        "observation.ee_pose.world": np.asarray(cur["ee_pose_world"], np.float32),
        "observation.ee_pose.robot": np.asarray(cur["ee_pose_robot"], np.float32),
        "action.ee_relative": encode(delta, nxt["opening_m"]),
        "observation.relative_state": np.r_[
            encode(past, prev["opening_m"]), cur["opening_m"]
        ].astype(np.float32),
        "observation.gripper_width_measured": np.array(
            [cur["width_measured"]], np.float32
        ),
        "observation.sensor_time": np.array(
            [cur["time_s"], cur["device_s"], cur["arrival_s"]], np.float64
        ),
        "action.time_offset": np.array([nxt["time_s"] - cur["time_s"]], np.float32),
        "observation.ik_error": np.array(
            [
                cur["ik_residual"]["position_error_m"],
                cur["ik_residual"]["rotation_error_rad"],
            ],
            np.float32,
        ),
    }


def verify(root, repo_id, mode):
    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    previous = None
    for i in range(len(ds)):
        row = ds[i]
        state, action = row["observation.state"].numpy(), row["action"].numpy()
        if (
            state.shape != ((10,) if mode == "ee" else (7,))
            or not np.isfinite([state, action]).all()
        ):
            raise ValueError("State/action shape or finiteness error")
        ep = int(row["episode_index"])
        if (
            previous
            and previous[0] == ep
            and not np.allclose(previous[1], state, atol=1e-6)
        ):
            raise ValueError("Next-frame action mismatch")
        matrix(row["observation.ee_pose"].numpy())
        if not row["observation.images.wrist"].isfinite().all():
            raise ValueError("Invalid decoded RGB")
        previous = ep, action
    count = 0
    for path in (Path(root) / "videos/observation.images.wrist").rglob("*.mp4"):
        with av.open(str(path)) as video:
            count += sum(1 for _ in video.decode(video=0))
    if count != len(ds):
        raise ValueError("Video frame count does not match labels")
    return {
        "format": "v3.0",
        "episodes": ds.num_episodes,
        "frames": len(ds),
        "all_frames_read_back": True,
        "decoded_video_frames": count,
    }


def export_pair(
    manifests,
    output,
    repo_id,
    *,
    obs_step=3,
    with_world=False,
    include_depth=True,
    allow_provisional=False,
    expected_episodes=None,
):
    if CODEBASE_VERSION != "v3.0":
        raise ValueError("LeRobot v3 required")
    docs = [json.loads(Path(p).read_text()) for p in manifests]
    if not docs or obs_step < 1:
        raise ValueError("Provide manifests and positive observation step")
    segments = [(doc, seg) for doc in docs for seg in doc["segments"]]
    if not segments or (
        expected_episodes is not None and len(segments) != expected_episodes
    ):
        raise ValueError(
            f"Expected {expected_episodes} accepted demonstrations, found {len(segments)}"
        )
    first = docs[0]
    verified = all(
        d["calibration_verified"] and d["robot_registration_verified"] for d in docs
    )
    if not verified and not allow_provisional:
        raise ValueError("Provisional manifests require --allow-provisional")
    for d in docs:
        if (
            d["calibration"] != first["calibration"]
            or d["robot_profile"] != first["robot_profile"]
        ):
            raise ValueError(
                "Collection requires identical calibration and robot profiles"
            )
        if d["contract"] != CONTRACT or (d["fps"], d["calibration"]["resolution"]) != (
            first["fps"],
            first["calibration"]["resolution"],
        ):
            raise ValueError("Mismatched contracts, rates or resolutions")
        if d.get("joint_label_source") != "IK-derived; not measured robot feedback":
            raise ValueError("A joint-labelled manifest is required for paired export")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / m for m in ("ee", "joint7")]
    stages = [p.with_name(p.name + ".partial") for p in destinations]
    if any(p.exists() for p in destinations + stages):
        raise FileExistsError(
            "Use a fresh output: datasets/partial stages are never overwritten"
        )
    w, h = first["calibration"]["resolution"]
    reports = {}
    for mode, stage in zip(("ee", "joint7"), stages):
        rid = repo_id + "-" + mode
        ds = LeRobotDataset.create(
            repo_id=rid,
            root=stage,
            fps=first["fps"],
            features=features(w, h, mode, with_world),
            robot_type="umi_yam_" + mode,
            video_backend="pyav",
            vcodec="h264",
            image_writer_threads=2,
        )
        mappings = []
        try:
            for ep, (doc, segment) in enumerate(segments):
                raw = Path(doc["raw_root"])
                fs = segment["frames"]
                extras = stage / "extras" / f"episode_{ep:06d}"
                extras.mkdir(parents=True)
                # Original sensor logs retain full-rate pose/IMU, RGB/depth
                # timestamp domains, counters, arrival times and raw paths.
                shutil.copyfile(raw / "samples.jsonl", extras / "samples.jsonl")
                shutil.copyfile(raw / "metadata.json", extras / "metadata.json")
                (extras / "manifest.json").write_text(
                    json.dumps(doc | {"segments": [segment]}, indent=2)
                )
                mappings.append(
                    {
                        "episode": ep,
                        "raw_root": str(raw),
                        "source_sha256": doc["source_sha256"],
                        "source_frames": [f["source_frame"] for f in fs],
                        "calibration_verified": doc["calibration_verified"],
                    }
                )
                for i in range(len(fs) - 1):
                    f = fs[i]
                    bgr = cv2.imread(str(raw / f["rgb_path"]))
                    if bgr is None:
                        raise ValueError("Missing wrist RGB")
                    values = frame_values(fs, i, mode, obs_step)
                    values["observation.images.wrist"] = cv2.cvtColor(
                        bgr, cv2.COLOR_BGR2RGB
                    )
                    if with_world:
                        im = cv2.imread(str(raw / f["world_path"]))
                        if im is None:
                            raise ValueError("Missing world RGB")
                        values["observation.images.world"] = cv2.cvtColor(
                            im, cv2.COLOR_BGR2RGB
                        )
                    values["task"] = doc["calibration"]["task"]
                    ds.add_frame(values)
                    if include_depth:
                        path = raw / f["depth_path"]
                        loaded = np.load(path, allow_pickle=False)
                        if isinstance(loaded, np.lib.npyio.NpzFile):
                            depth = loaded["depth"]
                            loaded.close()
                        else:
                            depth = loaded
                        if depth.dtype != np.uint16 or depth.shape != (h, w):
                            raise ValueError(
                                "Expected native uint16 depth with matching dimensions"
                            )
                        (extras / "depth").mkdir(exist_ok=True)
                        np.savez_compressed(
                            extras / "depth" / f"{i:08d}.npz", depth=depth
                        )
                ds.save_episode()
            contract = {
                "contract": CONTRACT,
                "mode": mode,
                "fps": first["fps"],
                "obs_step_frames": obs_step,
                "calibration_verified": verified,
                "hardware_validated": False,
                "joint_label_source": "IK-derived; not measured robot feedback",
                "state_names": EE_NAMES if mode == "ee" else JOINT_NAMES,
                "action_semantics": "next captured state, last observation omitted; relative action is inv(T_now) @ T_next",
                "gripper_semantics": "physical aperture in metres, not normalized motor command",
                "world_semantics": "T265 session world, not a registered robot-base absolute measurement",
                "robot_semantics": "each segment rigidly retargeted from profile initial_joints_rad with registered/declared tool transform",
                "depth_semantics": "extras depth is native unaligned Z16; multiply metadata depth_scale_m; use separate RGB/depth intrinsics/extrinsics",
                "source_mapping": mappings,
                "robot_models": [d["robot_model"] for d in docs],
            }
            (stage / "meta/umi_contract.json").write_text(
                json.dumps(contract, indent=2) + "\n"
            )
        except BaseException as e:
            (stage / "FAILED.txt").write_text(str(e))
            raise
        finally:
            ds.finalize()
            if ds.image_writer:
                ds.image_writer.stop()
        report = verify(stage, rid, mode)
        (stage / "verification.json").write_text(json.dumps(report, indent=2))
        reports[mode] = report
    if reports["ee"] != reports["joint7"]:
        raise ValueError("Paired dataset counts do not match")
    for stage, dest in zip(stages, destinations):
        stage.rename(dest)
    (output / "paired-verification.json").write_text(json.dumps(reports, indent=2))
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo-id", default="local/yumi-10")
    p.add_argument("--obs-step", type=int, default=3)
    p.add_argument("--with-world", action="store_true")
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--allow-provisional", action="store_true")
    p.add_argument("--expected-episodes", type=int)
    a = p.parse_args()
    print(
        json.dumps(
            export_pair(
                a.manifests,
                a.output,
                a.repo_id,
                obs_step=a.obs_step,
                with_world=a.with_world,
                include_depth=not a.no_depth,
                allow_provisional=a.allow_provisional,
                expected_episodes=a.expected_episodes,
            ),
            indent=2,
        )
    )
