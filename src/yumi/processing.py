import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    check_motion,
    interpolate_pose,
    pose_matrix,
    state,
    synchronize_streams,
    transform,
)
from .markers import Markers


def load_raw(root):
    root = Path(root)
    if not (root / "COMPLETE").exists() or (root / "FAILED.txt").exists():
        raise ValueError("Raw capture is incomplete or failed")
    meta = json.loads((root / "metadata.json").read_text())
    rows = [
        json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()
    ]
    return meta, rows


def prepare(root, output, calibration=None):
    root, output = Path(root), Path(output)
    meta, rows = load_raw(root)
    c = calibration or meta["config"]
    if not c["calibration_verified"] or not c["timing_verified"]:
        raise ValueError(
            "Measured spatial and temporal calibration must be verified before export"
        )
    if (
        c["rgb_serial"] != meta["config"]["rgb_serial"]
        or c["tracking_serial"] != meta["config"]["tracking_serial"]
    ):
        raise ValueError("Calibration serials do not match capture")
    ext = transform(c["T_tracker_tcp"])
    images = [r for r in rows if r["kind"] == "image"]
    poses = [r for r in rows if r["kind"] == "pose"]
    depths = [r for r in rows if r["kind"] == "depth"]
    if depths:
        td = np.asarray([r["device_s"] for r in depths])
        if np.any(np.diff(td) <= 0):
            raise ValueError("Depth timestamp reset or duplicate")
        for row in images:
            i = int(np.argmin(abs(td - row["device_s"])))
            if depths[i]["domain"] != row["domain"] or abs(
                td[i] - row["device_s"]
            ) > c.get("max_depth_skew_s", 0.025):
                raise ValueError("Missing nearby depth exposure")
            row["depth_path"] = depths[i]["depth_path"]
    for group in (images, poses):
        if len(group) < 10:
            raise ValueError("Too few camera/tracker samples")
        if len({r["domain"] for r in group}) != 1:
            raise ValueError("Timestamp domain changed")
        if np.any(np.diff([r["frame_number"] for r in group]) <= 0):
            raise ValueError("Duplicate / reset frame number")
    ti, tp, image_clock, pose_clock = synchronize_streams(images, poses)
    ti += c["camera_time_offset_s"]
    for report in (image_clock, pose_clock):
        if report["arrival_jitter_p95_s"] > c["max_clock_jitter_s"]:
            raise ValueError(f"Excessive USB timing jitter: {report}")
    start = max(ti[0], tp[0]) + c["warmup_s"]
    end = min(ti[-1], tp[-1]) - 0.1
    grid = np.arange(start, end, 1 / c["fps"])
    if len(grid) < 3:
        raise ValueError("Episode too short after warmup")
    right = np.searchsorted(ti, grid).clip(1, len(ti) - 1)
    idx = np.where(abs(ti[right] - grid) < abs(ti[right - 1] - grid), right, right - 1)
    if len(set(idx)) != len(idx):
        raise ValueError("Would duplicate RGB frames; lower output FPS or fix capture")
    skew = ti[idx] - grid
    if max(abs(skew)) > c["max_image_skew_s"]:
        raise ValueError("RGB gap / output grid misalignment")
    p = np.asarray([r["position"] for r in poses])
    q = np.asarray([r["quaternion_xyzw"] for r in poses])
    conf = np.asarray([r["tracker_confidence"] for r in poses])
    # Check every raw pose during the selected interval, including between video frames.
    selected = (tp >= grid[0] - 0.03) & (tp <= grid[-1] + 0.03)
    if np.min(conf[selected]) < c["min_tracker_confidence"]:
        raise ValueError("Tracking loss inside episode")
    raw_mats = np.asarray([pose_matrix(a, b) for a, b in zip(p[selected], q[selected])])
    check_motion(
        tp[selected], raw_mats, c["max_speed_m_s"], c["max_angular_speed_rad_s"]
    )
    detector = Markers(c["markers"], meta["rgb_intrinsics"])
    matrices, widths, errors = [], [], []
    for n, i in enumerate(idx):
        # Pose evaluated at actual image time; nominal LeRobot grid has bounded skew.
        a = (
            interpolate_pose(
                tp, p, q, conf, ti[i], c["max_pose_gap_s"], c["min_tracker_confidence"]
            )
            @ ext
        )
        bgr = cv2.imread(str(root / images[i]["rgb_path"]))
        if bgr is None:
            raise ValueError("Missing RGB file")
        try:
            width, error, _ = detector.width(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except ValueError as e:
            raise ValueError(f"Image {images[i]['index']}: {e}") from e
        matrices.append(a)
        widths.append(width)
        errors.append(error)
    matrices = np.asarray(matrices)
    check_motion(ti[idx], matrices, c["max_speed_m_s"], c["max_angular_speed_rad_s"])
    if np.any(abs(np.diff(widths)) / np.diff(ti[idx]) > c["max_width_speed_m_s"]):
        raise ValueError("Aperture discontinuity")
    # Full SE(3) episode origin: no Euler angles and no axis-sign guesswork.
    origin = np.linalg.inv(matrices[0])
    matrices = origin @ matrices
    states = np.asarray(
        [state(a, b) for a, b in zip(matrices, widths)], dtype=np.float32
    )
    for i in range(1, len(states)):
        if np.dot(states[i - 1, 3:7], states[i, 3:7]) < 0:
            states[i, 3:7] *= -1
    report = {
        "image_clock": image_clock,
        "pose_clock": pose_clock,
        "max_image_skew_s": float(max(abs(skew))),
        "max_marker_reprojection_px": float(max(errors)),
        "frames": len(states) - 1,
    }
    manifest = {
        "raw_root": str(root.resolve()),
        "fps": c["fps"],
        "task": c["task"],
        "config": c,
        "metadata": meta,
        "report": report,
        "action_semantics": "next sampled absolute TCP pose in episode-initial TCP frame; xyzw quaternion; aperture metres; last observation omitted",
        "source_sha256": hashlib.sha256(
            (root / "samples.jsonl").read_bytes()
        ).hexdigest(),
        "samples": [],
    }
    for n, i in enumerate(idx[:-1]):
        manifest["samples"].append(
            {
                "rgb_path": images[i]["rgb_path"],
                "depth_path": images[i]["depth_path"],
                "state": states[n].tolist(),
                "action": states[n + 1].tolist(),
                "sensor_time_s": float(ti[i] - grid[0]),
                "image_skew_s": float(skew[n]),
                "marker_error_px": errors[n],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    return report
