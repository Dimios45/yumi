"""One strict raw-to-demonstration path for Cartesian and IK-derived labels.

Every retained frame has a bracketed pose and an in-range aperture. Raw pose
corrections are rejected even at confidence 3. Invalid intervals split episodes;
an action can never cross them. No calibration flags are changed here.
"""

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
from .processing import load_raw
from .umi import encode_pose

CONTRACT = "yumi.demonstration.ee_joint7.v1"


def complete_episode_segment(doc, max_edge_trim_frames=6):
    """Reject fragmented demonstrations and large missing starts/finishes."""
    if len(doc["segments"]) != 1:
        raise ValueError(f"Expected one clean segment: {doc['report']}")
    frames = doc["segments"][0]["frames"]
    if not frames or max_edge_trim_frames < 0:
        raise ValueError("Empty segment or invalid permitted edge trimming")
    if (
        frames[0]["source_frame"] > max_edge_trim_frames
        or doc["report"]["images"] - 1 - frames[-1]["source_frame"]
        > max_edge_trim_frames
    ):
        raise ValueError(
            "Valid segment omits too much of the task; repeat the complete demonstration"
        )
    if any(b["source_frame"] != a["source_frame"] + 1 for a, b in pairwise(frames)):
        raise ValueError("Noncontiguous demonstration frames")
    return doc["segments"][0]


def validate_stream(rows, name, minimum=2):
    if len(rows) < minimum:
        raise ValueError(f"Too few {name} samples")
    if len({r["domain"] for r in rows}) != 1:
        raise ValueError(f"{name} timestamp domain changed")
    t = np.asarray([r["device_s"] for r in rows], float)
    n = np.asarray([r["frame_number"] for r in rows])
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or np.any(np.diff(n) <= 0):
        raise ValueError(f"{name} timestamp/counter duplicate or reset")
    return t


def bounded_widths(times, widths, max_gap_s=0.0):
    """Optionally fill a short interior gap; never extrapolate or hold endpoints."""
    t, w = np.asarray(times, float), np.asarray(widths, float).copy()
    if not np.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError("Nonnegative finite aperture interpolation gap required")
    measured = np.isfinite(w)
    indices = np.flatnonzero(measured)
    for left, right in pairwise(indices):
        if right > left + 1 and t[right] - t[left] <= max_gap_s:
            w[left + 1 : right] = np.interp(
                t[left + 1 : right], t[[left, right]], w[[left, right]]
            )
    return w, measured


def prepare_episode(
    root,
    config=None,
    *,
    allow_provisional=False,
    min_seconds=2.0,
    max_width_gap_s=0.0,
    require_depth=True,
    with_world=False,
):
    root = Path(root).resolve()
    meta, rows = load_raw(root)
    c = config or meta["config"]
    verified = bool(c.get("calibration_verified") and c.get("timing_verified"))
    if not verified and not allow_provisional:
        raise ValueError(
            "Spatial/timing calibration unverified; --allow-provisional permits labelled research data only"
        )
    ext = transform(c.get("T_tracker_tcp"))
    if not np.isfinite(min_seconds) or min_seconds <= 0:
        raise ValueError("Positive finite minimum segment duration required")
    for key in ("rgb_serial", "tracking_serial", "resolution"):
        if c[key] != meta["config"][key]:
            raise ValueError(f"Calibration identity mismatch: {key}")
    images = [r for r in rows if r["kind"] == "image"]
    poses = [r for r in rows if r["kind"] == "pose"]
    validate_stream(images, "image", 10)
    validate_stream(poses, "pose", 10)
    ti, tp, ic, pc = synchronize_streams(images, poses)
    ti += float(c["camera_time_offset_s"])
    for report in (ic, pc):
        if (
            report["alignment"] != "shared_sdk_global_time"
            and report["arrival_jitter_p95_s"] > c["max_clock_jitter_s"]
        ):
            raise ValueError("Unstable private-clock alignment")
    position = np.asarray([r["position"] for r in poses])
    quat = np.asarray([r["quaternion_xyzw"] for r in poses])
    conf = np.asarray([r["tracker_confidence"] for r in poses])
    mats = np.asarray([pose_matrix(p, q) @ ext for p, q in zip(position, quat)])
    # Precompute all bad high-rate intervals, including brief events hidden by
    # RGB resampling. A correction cannot be repaired by smoothing its labels.
    bad = np.zeros(len(tp) - 1, bool)
    for i in range(len(bad)):
        bad[i] = (
            min(conf[i : i + 2]) < c["min_tracker_confidence"]
            or tp[i + 1] - tp[i] > c["max_pose_gap_s"]
        )
        try:
            check_motion(
                tp[i : i + 2],
                mats[i : i + 2],
                c["max_speed_m_s"],
                c["max_angular_speed_rad_s"],
            )
        except ValueError:
            bad[i] = True
    detector = Markers(c["markers"], meta["rgb_intrinsics"])
    frames, widths = [], []
    for i, row in enumerate(images):
        f = {
            "source_frame": i,
            "rgb_path": row["rgb_path"],
            "time_s": float(ti[i]),
            "device_s": row["device_s"],
            "arrival_s": row["arrival_s"],
            "reasons": [],
        }
        try:
            f["T_world_tcp"] = (
                interpolate_pose(
                    tp,
                    position,
                    quat,
                    conf,
                    ti[i],
                    c["max_pose_gap_s"],
                    c["min_tracker_confidence"],
                )
                @ ext
            ).tolist()
            j = np.searchsorted(tp, ti[i], side="right")
            if bad[j - 1]:
                raise ValueError("Raw pose discontinuity/confidence loss")
        except ValueError as e:
            f["reasons"].append(str(e))
        bgr = cv2.imread(str(root / row["rgb_path"]))
        if bgr is None:
            raise ValueError(f"Missing RGB file: {row['rgb_path']}")
        try:
            width, _, _ = detector.width(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            f["width_error"] = None
        except ValueError as e:
            width = np.nan
            f["width_error"] = str(e)
        frames.append(f)
        widths.append(width)
    widths, measured = bounded_widths(ti, widths, max_width_gap_s)
    for i, f in enumerate(frames):
        # A bad geometric/calibration measurement is never repaired by width
        # interpolation. Only entirely missing detections may be filled.
        error = f["width_error"]
        if error and error != "No markers":
            widths[i] = np.nan
        f["width_measured"] = bool(measured[i])
        if np.isfinite(widths[i]):
            f["opening_m"] = float(widths[i])
        else:
            f["reasons"].append(f["width_error"])
        if i:
            left = max(0, np.searchsorted(tp, ti[i - 1], side="right") - 1)
            right = min(len(bad), np.searchsorted(tp, ti[i], side="right"))
            reason = None
            if np.any(bad[left:right]):
                reason = "Raw pose discontinuity/confidence loss between RGB frames"
            elif ti[i] - ti[i - 1] > 1.5 / c["fps"]:
                reason = "RGB gap"
            elif (
                np.isfinite(widths[i - 1 : i + 1]).all()
                and abs(widths[i] - widths[i - 1]) / (ti[i] - ti[i - 1])
                > c["max_width_speed_m_s"]
            ):
                reason = "Aperture discontinuity"
            if reason:
                frames[i - 1]["reasons"].append(reason)
                f["reasons"].append(reason)
    for kind, required in (("depth", require_depth), ("world", with_world)):
        group = [r for r in rows if r["kind"] == kind]
        if not required:
            continue
        times = validate_stream(group, kind)
        # Depth belongs to the same D405. A separate world camera must expose
        # SDK global time; matching two private epochs is not synchronization.
        if kind == "world" and group[0]["domain"] != "timestamp_domain.global_time":
            raise ValueError("World camera needs SDK global timestamps")
        for row, f in zip(images, frames):
            j = int(np.argmin(abs(times - row["device_s"])))
            limit = c.get(f"max_{kind}_skew_s", 0.025)
            if (
                group[j]["domain"] != row["domain"]
                or abs(times[j] - row["device_s"]) > limit
            ):
                f["reasons"].append(f"Missing synchronized {kind}")
                continue
            key = "depth_path" if kind == "depth" else "rgb_path"
            path = group[j][key]
            if not (root / path).is_file():
                raise ValueError(f"Missing {kind} file: {path}")
            f[f"{kind}_path"] = path
    segments, current = [], []
    for f in frames + [{"reasons": ["end"]}]:
        if not f["reasons"]:
            current.append(f)
        else:
            if (
                len(current) > 1
                and current[-1]["time_s"] - current[0]["time_s"] >= min_seconds
            ):
                # Bound nominal-vs-exposure timing before constructing fixed-FPS labels.
                t = np.asarray([x["time_s"] for x in current])
                skew = t - (t[0] + np.arange(len(t)) / c["fps"])
                if max(abs(skew)) <= c["max_image_skew_s"]:
                    segments.append(current)
            current = []
    docs = []
    for segment in segments:
        inv = np.linalg.inv(segment[0]["T_world_tcp"])
        for f in segment:
            f["ee_pose"] = encode_pose(inv @ f["T_world_tcp"], f["opening_m"]).tolist()
            f["ee_pose_world"] = encode_pose(f["T_world_tcp"], f["opening_m"]).tolist()
        docs.append({"frames": segment})
    reasons = Counter(reason for f in frames for reason in set(f["reasons"]))
    return {
        "contract": CONTRACT,
        "raw_root": str(root),
        "calibration": c,
        "calibration_verified": verified,
        "allow_provisional": allow_provisional,
        "source_sha256": hashlib.sha256(
            (root / "samples.jsonl").read_bytes()
        ).hexdigest(),
        "metadata": meta,
        "fps": c["fps"],
        "segments": docs,
        "report": {
            "images": len(images),
            "valid_frames": sum(not f["reasons"] for f in frames),
            "raw_bad_intervals": int(bad.sum()),
            "invalid_reasons": dict(reasons),
            "segments": len(docs),
            "image_clock": ic,
            "pose_clock": pc,
            "max_width_gap_s": max_width_gap_s,
            "interpolated_width_frames": int((np.isfinite(widths) & ~measured).sum()),
        },
    }


def write_manifest(doc, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
