"""SE(3), clock fitting and strictly bracketed pose interpolation."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("Expected finite 4x4 transform")
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous row")
    if not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(a[:3, :3]), 1
    ):
        raise ValueError("Rotation must be in SO(3)")
    return a


def pose_matrix(p, q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.02:
        raise ValueError("Invalid quaternion")
    a = np.eye(4)
    a[:3, :3] = Rotation.from_quat(q).as_matrix()
    a[:3, 3] = p
    return transform(a)


def state(a, width):
    return np.r_[a[:3, 3], Rotation.from_matrix(a[:3, :3]).as_quat(), width]


def fit_clock(device, arrival):
    """Fit device seconds to host monotonic seconds; lower-envelope USB delay.

    This removes clock offset/drift, NOT unknown sensor latency. Keep residuals
    and calibrate the camera-minus-pose time offset separately.
    """
    x, y = np.asarray(device, float), np.asarray(arrival, float)
    if len(x) < 10 or not np.isfinite([x, y]).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Clock reset, duplicate timestamp or too few samples")
    origin = x[0]
    x = x - origin
    slope = np.polyfit(x, y - y[0], 1)[0]
    if not 0.995 < slope < 1.005:
        raise ValueError(f"Implausible clock rate {slope}")
    offset = np.quantile(y - slope * x, 0.05)
    mapped = slope * x + offset
    residual = y - mapped
    return mapped, {
        "rate": float(slope),
        "arrival_jitter_p95_s": float(
            np.quantile(residual, 0.95) - np.quantile(residual, 0.05)
        ),
    }


def interpolate_pose(
    times, positions, quaternions, confidence, query, max_gap=0.025, min_confidence=3
):
    t = np.asarray(times)
    j = int(np.searchsorted(t, query, side="right"))
    if j == 0 or j == len(t):
        raise ValueError("Pose extrapolation forbidden")
    i = j - 1
    if t[j] - t[i] > max_gap or min(confidence[i], confidence[j]) < min_confidence:
        raise ValueError("Low confidence or pose gap")
    u = (query - t[i]) / (t[j] - t[i])
    p = (1 - u) * positions[i] + u * positions[j]
    q = Slerp(t[i : j + 1], Rotation.from_quat(quaternions[i : j + 1]))(
        [query]
    ).as_quat()[0]
    return pose_matrix(p, q)


def check_motion(times, matrices, max_speed, max_angular_speed):
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Non-monotonic samples")
    v = np.linalg.norm(np.diff(matrices[:, :3, 3], axis=0), axis=1) / dt
    r = Rotation.from_matrix(matrices[:, :3, :3])
    w = (r[:-1].inv() * r[1:]).magnitude() / dt
    if np.any(v > max_speed) or np.any(w > max_angular_speed):
        raise ValueError("Pose jump / excessive motion; reject episode")


def synchronize_streams(images, poses):
    """Use a common origin for SDK-global timestamps; fit only private clocks.

    Independently zeroing global-time streams would incorrectly remove their
    relative acquisition times and bake USB delivery delay into alignment.
    """
    arrays = []
    reports = []
    domains = []
    for rows in (images, poses):
        domain = {r["domain"] for r in rows}
        if len(domain) != 1:
            raise ValueError("Missing or changing timestamp domain")
        domains.append(next(iter(domain)))
        device = np.asarray([r["device_s"] for r in rows], float)
        arrival = np.asarray([r["arrival_s"] for r in rows], float)
        mapped, report = fit_clock(device, arrival)
        arrays.append((device, mapped))
        reports.append(report)
    if domains == ["timestamp_domain.global_time"] * 2:
        origin = min(arrays[0][0][0], arrays[1][0][0])
        mapped = [a[0] - origin for a in arrays]
        for report in reports:
            report["alignment"] = "shared_sdk_global_time"
    elif any(d == "timestamp_domain.global_time" for d in domains):
        raise ValueError(
            "Mixed global/private clocks; configure a common domain before calibration"
        )
    else:
        mapped = [a[1] for a in arrays]
        for report in reports:
            report["alignment"] = "independent_device_clock_fit"
    return mapped[0], mapped[1], reports[0], reports[1]
