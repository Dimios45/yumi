"""Audit and synchronize persistent LeRobot capture sessions without fake labels."""

import hashlib
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    check_motion,
    interpolate_pose,
    pose_matrix,
    synchronize_streams,
    transform,
)
from .markers import Markers
from .umi import CONTRACT, make_chunks


def stream_report(rows):
    result = {}
    for kind in sorted({r["kind"] for r in rows}):
        group = [r for r in rows if r["kind"] == kind]
        t = np.array([r["device_s"] for r in group])
        n = np.array([r["frame_number"] for r in group])
        result[kind] = {
            "count": len(group),
            "max_gap_s": float(np.diff(t).max()) if len(t) > 1 else None,
            "nonincreasing_timestamps": int((np.diff(t) <= 0).sum()),
            "counter_gaps": int(np.maximum(np.diff(n) - 1, 0).sum()),
            "counter_resets_or_duplicates": int((np.diff(n) <= 0).sum()),
        }
    return result


def video_frames(root):
    for path in sorted((root / "videos/observation.images.d405").rglob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        index = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield (
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    str(path.relative_to(root)),
                    index,
                )
                index += 1
        finally:
            cap.release()


def prepare_session(
    root,
    output,
    calibration=None,
    history=2,
    horizon=16,
    allow_provisional=False,
    task=None,
):
    root, output = Path(root).resolve(), Path(output).resolve()
    if history < 1 or horizon < 1:
        raise ValueError("Positive history and horizon required")
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if not (root / "COMPLETE").exists() or (root / "FAILED.txt").exists():
        raise ValueError("Session is incomplete or failed")
    meta = json.loads((root / "session.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected LeRobot v3")
    c = calibration or meta["config"]
    ext = transform(c["T_tracker_tcp"])
    for key in ["rgb_serial", "tracking_serial", "resolution"]:
        if c[key] != meta["config"][key]:
            raise ValueError(f"Calibration mismatch: {key}")
    verified = bool(c.get("calibration_verified") and c.get("timing_verified"))
    output.mkdir(parents=True)
    report = {
        "source": str(root),
        "contract": CONTRACT,
        "calibration_verified": verified,
        "allow_provisional": allow_provisional,
        "training_ready": False,
        "episodes": [],
        "note": "Valid chunks certify software gates only, not physical calibration or robot feasibility.",
    }
    detector = Markers(c["markers"], meta["rgb_intrinsics"])
    videos = iter(video_frames(root))
    total = 0
    episode_docs = []
    for ex in sorted((root / "extras").glob("episode_*")):
        rows = [json.loads(l) for l in (ex / "samples.jsonl").open()]
        images = [r for r in rows if r["kind"] == "image"]
        poses = [r for r in rows if r["kind"] == "pose"]
        audit = {
            "episode": ex.name,
            "streams": stream_report(rows),
            "invalid_reasons": {},
            "valid_frames": 0,
            "chunks": 0,
        }
        clock_error = None
        try:
            if len(images) < 10 or len(poses) < 10:
                raise ValueError("Too few samples")
            ti, tp, ic, pc = synchronize_streams(images, poses)
            ti += float(c["camera_time_offset_s"])
            audit["clocks"] = {"image": ic, "pose": pc}
            # Arrival jitter alone is not physical exposure jitter for shared SDK global time.
            if (
                ic["alignment"] != "shared_sdk_global_time"
                and max(ic["arrival_jitter_p95_s"], pc["arrival_jitter_p95_s"])
                > c["max_clock_jitter_s"]
            ):
                raise ValueError("Unstable private-clock mapping")
            if any(
                v["nonincreasing_timestamps"] or v["counter_resets_or_duplicates"]
                for k, v in audit["streams"].items()
                if k in ("pose", "image")
            ):
                raise ValueError("Image/pose timestamp or counter reset")
        except ValueError as e:
            clock_error = str(e)
            ti = np.arange(len(images)) / c["fps"]
        p = np.array([r["position"] for r in poses])
        q = np.array([r["quaternion_xyzw"] for r in poses])
        conf = np.array([r["tracker_confidence"] for r in poses])
        frames = []
        errors = Counter()
        for i, row in enumerate(images):
            try:
                rgb, path, video_index = next(videos)
            except StopIteration as e:
                raise ValueError("Video shorter than sensor image log") from e
            f = {
                "frame_index": i,
                "video_path": path,
                "video_frame_index": video_index,
                "time_s": float(ti[i]),
                "device_s": row["device_s"],
                "valid": False,
            }
            reasons = []
            a = None
            width = None
            if clock_error:
                reasons.append(clock_error)
            else:
                try:
                    a = (
                        interpolate_pose(
                            tp,
                            p,
                            q,
                            conf,
                            ti[i],
                            c["max_pose_gap_s"],
                            c["min_tracker_confidence"],
                        )
                        @ ext
                    )
                except ValueError as e:
                    reasons.append(str(e))
            # Re-detect using the selected calibration: no zero filling or width extrapolation.
            try:
                width, _, _ = detector.width(rgb)
            except ValueError as e:
                reasons.append(str(e))
            if a is not None:
                f["T_world_tcp"] = a.tolist()
            if width is not None:
                f["opening_m"] = float(width)
            f["reasons"] = reasons
            f["valid"] = not reasons
            frames.append(f)
            total += 1
        # Reject both sides of a jump, including fast aperture discontinuities.
        for prev, cur in pairwise(frames):
            if "T_world_tcp" not in prev or "T_world_tcp" not in cur:
                continue
            try:
                dt = cur["time_s"] - prev["time_s"]
                if not clock_error:
                    lo = max(0, np.searchsorted(tp, prev["time_s"], side="right") - 1)
                    hi = min(
                        len(tp), np.searchsorted(tp, cur["time_s"], side="right") + 1
                    )
                    if np.any(conf[lo:hi] < c["min_tracker_confidence"]) or np.any(
                        np.diff(tp[lo:hi]) > c["max_pose_gap_s"]
                    ):
                        raise ValueError("Tracking loss between image frames")
                    check_motion(
                        tp[lo:hi],
                        np.array(
                            [
                                pose_matrix(pi, qi) @ ext
                                for pi, qi in zip(p[lo:hi], q[lo:hi])
                            ]
                        ),
                        c["max_speed_m_s"],
                        c["max_angular_speed_rad_s"],
                    )
                check_motion(
                    [prev["time_s"], cur["time_s"]],
                    np.array([prev["T_world_tcp"], cur["T_world_tcp"]]),
                    c["max_speed_m_s"],
                    c["max_angular_speed_rad_s"],
                )
                if dt > 1.5 / c["fps"]:
                    raise ValueError("Image gap")
                if (
                    "opening_m" in prev
                    and "opening_m" in cur
                    and abs(cur["opening_m"] - prev["opening_m"]) / dt
                    > c["max_width_speed_m_s"]
                ):
                    raise ValueError("Aperture discontinuity")
            except ValueError as e:
                for f in (prev, cur):
                    f["valid"] = False
                    f["reasons"].append(str(e))
        for f in frames:
            errors.update(set(f["reasons"]))
        chunks = (
            make_chunks(frames, history, horizon, c["fps"])
            if verified or allow_provisional
            else []
        )
        audit.update(
            valid_frames=sum(f["valid"] for f in frames),
            chunks=len(chunks),
            invalid_reasons=dict(errors),
        )
        report["episodes"].append(audit)
        doc = {
            "contract": CONTRACT,
            "source_root": str(root),
            "source_episode": ex.name,
            "fps": c["fps"],
            "history": history,
            "horizon": horizon,
            "task": task or c["task"],
            "calibration": c,
            "calibration_verified": verified,
            "frames": frames,
            "samples": chunks,
            "source_sha256": hashlib.sha256(
                (ex / "samples.jsonl").read_bytes()
            ).hexdigest(),
        }
        episode_docs.append(doc)
    if next(videos, None) is not None or total != info["total_frames"]:
        raise ValueError("Video/log/LeRobot metadata frame count mismatch")
    report["total_frames"] = total
    report["total_chunks"] = sum(e["chunks"] for e in report["episodes"])
    report["training_ready"] = verified and report["total_chunks"] > 0
    report["status"] = (
        "ready_for_export"
        if report["training_ready"]
        else "provisional_chunks"
        if report["total_chunks"]
        else "blocked_no_valid_training_chunks"
    )
    for doc in episode_docs:
        (output / f"{doc['source_episode']}.json").write_text(
            json.dumps(doc, indent=2, allow_nan=False) + "\n"
        )
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
