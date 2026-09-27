"""Offline calibration solvers with numerical residual reports."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import transform


def fit_width(distances_m, widths_m):
    x, y = np.asarray(distances_m, float), np.asarray(widths_m, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 6 or not np.isfinite([x, y]).all():
        raise ValueError("Provide >=6 measured separation/aperture pairs")
    if np.ptp(y) < 0.02 or np.ptp(x) < 0.01:
        raise ValueError("Insufficient aperture range")
    a, b = np.polyfit(x, y, 1)
    if not 0.2 < a < 5:
        raise ValueError("Implausible width gain; check marker size and units")
    residual = a * x + b - y
    return {
        "width_gain": float(a),
        "width_offset_m": float(b),
        "rms_m": float(np.sqrt(np.mean(residual**2))),
        "max_error_m": float(max(abs(residual))),
    }


def hand_eye(world_tracker, camera_board):
    """Stationary calibration board; return T_tracker_camera (AX=XB)."""
    a = np.asarray([transform(x) for x in world_tracker])
    b = np.asarray([transform(x) for x in camera_board])
    if len(a) != len(b) or len(a) < 12:
        raise ValueError("Need >=12 paired static poses")
    relative = Rotation.from_matrix(a[0, :3, :3].T @ a[:, :3, :3]).as_rotvec()
    if np.linalg.svd(relative, compute_uv=False)[1] < 0.3:
        raise ValueError("Rotate around at least two independent axes")
    r, t = cv2.calibrateHandEye(
        list(a[:, :3, :3]),
        list(a[:, :3, 3]),
        list(b[:, :3, :3]),
        list(b[:, :3, 3]),
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    x = np.eye(4)
    x[:3, :3] = r
    x[:3, 3] = t.ravel()
    transform(x)
    anchors = a @ x @ b
    pos = anchors[:, :3, 3]
    mean_r = Rotation.from_matrix(anchors[:, :3, :3]).mean()
    return {
        "T_tracker_camera": x.tolist(),
        "board_position_rms_m": float(
            np.sqrt(np.mean(np.sum((pos - pos.mean(axis=0)) ** 2, axis=1)))
        ),
        "board_rotation_rms_rad": float(
            np.sqrt(
                np.mean(
                    (
                        mean_r.inv() * Rotation.from_matrix(anchors[:, :3, :3])
                    ).magnitude()
                    ** 2
                )
            )
        ),
    }


def estimate_time_offset(
    camera_times, camera_rotations, tracker_times, tracker_rotations, search_s=0.15
):
    """Find correction ADDED to camera times from rotation-speed correlation.

    A rigid camera/tracker pair has the same angular-speed magnitude despite
    their different axes. Requires a stationary observed board and varied motion.
    Reports correlation/peak separation; never marks timing verified automatically.
    """
    ct, tt = np.asarray(camera_times), np.asarray(tracker_times)
    cr, tr = (
        Rotation.from_matrix(camera_rotations),
        Rotation.from_matrix(tracker_rotations),
    )
    if (
        len(ct) < 30
        or len(tt) < 100
        or np.any(np.diff(ct) <= 0)
        or np.any(np.diff(tt) <= 0)
    ):
        raise ValueError("Need monotonic samples spanning varied rotational motion")
    cv = (cr[:-1].inv() * cr[1:]).magnitude() / np.diff(ct)
    tv = (tr[:-1].inv() * tr[1:]).magnitude() / np.diff(tt)
    cm, tm = (ct[:-1] + ct[1:]) / 2, (tt[:-1] + tt[1:]) / 2
    keep = (cm - search_s > tm[0]) & (cm + search_s < tm[-1]) & (np.diff(ct) < 0.08)
    cm, cv = cm[keep], cv[keep]
    if len(cv) < 25 or np.std(cv) < 0.15:
        raise ValueError("Insufficient rotational excitation for timing calibration")
    offsets = np.linspace(-search_s, search_s, 301)
    scores = np.array(
        [np.corrcoef(cv, np.interp(cm + d, tm, tv))[0, 1] for d in offsets]
    )
    best = int(np.nanargmax(scores))
    distant = abs(offsets - offsets[best]) > 0.03
    separation = float(scores[best] - np.nanmax(scores[distant]))
    if best in (0, len(offsets) - 1) or scores[best] < 0.8 or separation < 0.02:
        raise ValueError(
            "Timing correlation ambiguous; collect nonperiodic varied rotations"
        )
    return {
        "camera_time_offset_s": float(offsets[best]),
        "correlation": float(scores[best]),
        "peak_separation": separation,
        "search_resolution_s": float(offsets[1] - offsets[0]),
    }


def calibrate_board(raw_root, marker_id, size_m, dictionary="DICT_4X4_50"):
    """Extract stationary-board observations from a dedicated calibration capture."""
    from pathlib import Path

    from .geometry import interpolate_pose, synchronize_streams
    from .markers import camera_matrix, rectify_points
    from .processing import load_raw

    meta, rows = load_raw(raw_root)
    images = [r for r in rows if r["kind"] == "image"]
    poses = [r for r in rows if r["kind"] == "pose"]
    if size_m <= 0:
        raise ValueError("Board marker size must be positive")
    ti, tp, image_clock, pose_clock = synchronize_streams(images, poses)
    limit = meta["config"].get("max_clock_jitter_s", 0.015)
    if any(r["arrival_jitter_p95_s"] > limit for r in (image_clock, pose_clock)):
        raise ValueError(
            "Capture timing jitter exceeds calibration gate; fix acquisition first"
        )
    intr = meta["rgb_intrinsics"]
    k = camera_matrix(intr)
    d = np.zeros(5)
    s = size_m / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params
    )
    times = []
    boards = []
    for timestamp, row in zip(ti, images):
        im = cv2.imread(str(Path(raw_root) / row["rgb_path"]), cv2.IMREAD_GRAYSCALE)
        corners, ids, _ = detector.detectMarkers(im)
        if ids is None:
            continue
        found = np.flatnonzero(ids.ravel() == marker_id)
        if len(found) != 1:
            continue
        points = corners[found[0]].reshape(4, 2).astype(float)
        if np.min(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)) < 40:
            continue
        points = rectify_points(points, intr)
        ok, rv, tv = cv2.solvePnP(obj, points, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tv[2, 0] <= 0:
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, k, d)
        if np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - points) ** 2, axis=1))) > 1.0:
            continue
        b = np.eye(4)
        b[:3, :3] = cv2.Rodrigues(rv)[0]
        b[:3, 3] = tv.ravel()
        times.append(timestamp)
        boards.append(b)
    from .geometry import pose_matrix

    tracker = np.array(
        [pose_matrix(r["position"], r["quaternion_xyzw"]) for r in poses]
    )
    boards = np.asarray(boards)
    if len(boards) < 30:
        raise ValueError("Too few large calibration-board detections")
    timing = estimate_time_offset(times, boards[:, :3, :3], tp, tracker[:, :3, :3])
    pp = np.array([r["position"] for r in poses])
    qq = np.array([r["quaternion_xyzw"] for r in poses])
    cc = np.array([r["tracker_confidence"] for r in poses])
    aa = []
    bb = []
    for timestamp, b in zip(times, boards):
        try:
            a = interpolate_pose(
                tp, pp, qq, cc, timestamp + timing["camera_time_offset_s"]
            )
        except ValueError:
            continue
        aa.append(a)
        bb.append(b)
    result = hand_eye(aa, bb)
    return (
        result
        | timing
        | {"image_clock": image_clock, "pose_clock": pose_clock}
        | {
            "paired_samples": len(aa),
            "T_world_tracker": [a.tolist() for a in aa],
            "T_camera_board": [b.tolist() for b in bb],
        }
    )


def capture_width_sample(config, opening_mm, samples_path, frames=60):
    """Capture one physically held jaw setting and append a traceable measurement."""
    import json
    import time
    from collections import Counter
    from pathlib import Path

    from .hardware import intrinsics, sdk
    from .markers import Markers

    if not np.isfinite(opening_mm) or opening_mm < 0 or frames < 20:
        raise ValueError(
            "Opening must be a nonnegative measured value; need >=20 frames"
        )
    path = Path(samples_path)
    image_mode = (
        config["markers"].get("width_method", "metric_pnp") == "image_calibrated"
    )
    feature_key = "marker_separation_px" if image_mode else "marker_separation_m"
    doc = (
        json.loads(path.read_text())
        if path.exists()
        else {feature_key: [], "jaw_opening_m": [], "captures": []}
    )
    if feature_key not in doc:
        raise ValueError("Use a new sample file for the selected width method")
    fingerprint = {
        "width_method": config["markers"].get("width_method", "metric_pnp"),
        "resolution": config["resolution"],
        "rgb_serial": config["rgb_serial"],
        "markers": {
            key: config["markers"][key] for key in ("dictionary", "ids", "size_m")
        },
    }
    if doc.get("calibration_identity", fingerprint) != fingerprint:
        raise ValueError("Sample file belongs to a different camera or marker geometry")
    doc["calibration_identity"] = fingerprint
    rs = sdk()
    pipe = rs.pipeline()
    c = rs.config()
    c.enable_device(config["rgb_serial"])
    w, h = config["resolution"]
    c.enable_stream(rs.stream.color, w, h, rs.format.rgb8, config["fps"])
    profile = pipe.start(c)
    values = []
    errors = []
    failures = Counter()
    last = -1
    try:
        detector = Markers(
            config["markers"], intrinsics(profile.get_stream(rs.stream.color))
        )
        for _ in range(30):
            pipe.wait_for_frames(3000)
        for _ in range(frames):
            frame = pipe.wait_for_frames(3000).get_color_frame()
            if frame.get_frame_number() <= last:
                failures["Duplicate frame"] += 1
                continue
            last = frame.get_frame_number()
            try:
                distance, error, _ = detector.measurement(
                    np.asanyarray(frame.get_data())
                )
                values.append(distance)
                errors.append(error)
            except ValueError as e:
                failures[str(e)] += 1
    finally:
        pipe.stop()
    if len(values) < 0.8 * frames:
        raise ValueError(
            f"Only {len(values)}/{frames} frames valid: {dict(failures)}. Improve marker visibility/geometry before fitting."
        )
    spread = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    spread_limit = 2.0 if image_mode else 0.001
    if spread > spread_limit:
        raise ValueError(
            f"Marker separation spread is {spread:.4f} {'pixels' if image_mode else 'metres'}; hold a fixed gauge and improve image quality"
        )
    capture = {
        "measured_jaw_opening_m": opening_mm / 1000,
        feature_key: float(np.median(values)),
        "valid_frames": len(values),
        "separation_p95_p05": spread,
        "separation_units": "pixels" if image_mode else "metres",
        "reprojection_p95_px": float(np.quantile(errors, 0.95)),
        "unix_s": time.time(),
    }
    doc[feature_key].append(capture[feature_key])
    doc["jaw_opening_m"].append(capture["measured_jaw_opening_m"])
    doc["captures"].append(capture)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
    temporary.replace(path)
    return capture


def fit_image_width(samples, max_error_m=0.001):
    """Fit fixed-camera parallel-jaw opening; reject poor fit, never extrapolate."""
    if not np.isfinite(max_error_m) or max_error_m <= 0:
        raise ValueError("Fit error limit must be finite and positive")
    x = np.asarray(samples["marker_separation_px"], float)
    y = np.asarray(samples["jaw_opening_m"], float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 6 or not np.isfinite([x, y]).all():
        raise ValueError("Need at least six measured opening settings")
    if len(np.unique(np.round(y, 5))) < 6 or np.ptp(y) < 0.02 or np.ptp(x) < 10:
        raise ValueError(
            "Need six distinct jaw settings spanning >=20 mm and >=10 pixels"
        )
    gain, offset = np.polyfit(x, y, 1)
    residual = gain * x + offset - y
    if gain <= 0 or max(abs(residual)) > max_error_m:
        raise ValueError(
            f"Image-to-opening fit exceeds {max_error_m * 1000:g} mm or is not increasing; check geometry and measurements"
        )
    return {
        "width_method": "image_calibrated",
        "image_width_calibration": {
            "gain_m_per_px": float(gain),
            "offset_m": float(offset),
            "separation_range_px": [float(min(x)), float(max(x))],
            "resolution": samples["calibration_identity"]["resolution"],
            "rms_m": float(np.sqrt(np.mean(residual**2))),
            "max_training_error_m": float(max(abs(residual))),
            "accepted_fit_error_limit_m": float(max_error_m),
            "validation_status": "provisional_pipeline_test"
            if max_error_m > 0.001
            else "fit_only_not_independently_validated",
        },
        "note": "Verify on independent held-out openings before approving calibration",
    }
