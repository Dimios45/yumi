"""Read-only raw recording audit; no calibration flags or samples are changed."""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .geometry import fit_clock
from .markers import Markers
from .processing import load_raw


def inspect_recording(root, config=None):
    meta, rows = load_raw(root)
    c = config or meta["config"]
    report = {"raw_root": str(Path(root).resolve()), "streams": {}}
    for kind in ("image", "depth", "pose", "gyro", "accel"):
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        t = np.asarray([r["device_s"] for r in group])
        host = np.asarray([r["arrival_s"] for r in group])
        counter = np.asarray([r["frame_number"] for r in group])
        gaps = int(np.maximum(np.diff(counter) - 1, 0).sum())
        s = {
            "samples": len(group),
            "span_s": float(np.ptp(t)),
            "counter_gaps": gaps,
            "missing_fraction": gaps / (len(group) + gaps),
            "duplicates": int(np.sum(np.diff(counter) == 0)),
            "counter_resets": int(np.sum(np.diff(counter) < 0)),
            "nonincreasing_timestamps": int(np.sum(np.diff(t) <= 0)),
            "domains": sorted({r["domain"] for r in group}),
        }
        if len(t) > 1:
            s["rate_hz"] = (
                float((len(t) - 1) / (t[-1] - t[0])) if t[-1] > t[0] else None
            )
            s["max_gap_s"] = float(max(np.diff(t)))
        try:
            _, s["clock"] = fit_clock(t, host)
        except ValueError as e:
            s["clock_error"] = str(e)
        if kind == "pose":
            conf = np.asarray([r["tracker_confidence"] for r in group])
            s["confidence_counts"] = dict(Counter(conf.tolist()))
            s["confidence_3_fraction"] = float(np.mean(conf == 3))
            xyz = np.asarray([r["position"] for r in group])
            s["position_extent_m"] = np.ptp(xyz, axis=0).tolist()
            s["start_end_distance_m"] = float(np.linalg.norm(xyz[-1] - xyz[0]))
            # Endpoints may be deliberately different: not a measured drift error.
            good = conf == 3
            boundaries = np.flatnonzero(np.diff(np.r_[False, good, False]))
            s["confidence_3_intervals_s"] = [
                [float(t[a] - t[0]), float(t[b - 1] - t[0])]
                for a, b in zip(boundaries[::2], boundaries[1::2])
            ]
        report["streams"][kind] = s
    if c.get("markers", {}).get("size_m"):
        detector = Markers(c["markers"], meta["rgb_intrinsics"])
        counts = Counter()
        reasons = Counter()
        values = []
        errors = []
        examples = {}
        images = [r for r in rows if r["kind"] == "image"]
        for row in images:
            image = cv2.imread(str(Path(root) / row["rgb_path"]))
            if image is None:
                reasons["Missing RGB file"] += 1
                continue
            _corners, ids, _ = detector.detector.detectMarkers(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            )
            found = [] if ids is None else ids.ravel().tolist()
            for marker in c["markers"]["ids"]:
                counts[str(marker)] += marker in found
            try:
                distance, error, _ = detector.measure(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )
                values.append(distance)
                errors.append(error)
            except ValueError as e:
                reason = str(e)
                reasons[reason] += 1
                examples.setdefault(reason, row["rgb_path"])
        report["markers"] = {
            "size_m": c["markers"]["size_m"],
            "images_checked": len(images),
            "individual_detection_counts": dict(counts),
            "metric_pair_accepted": len(values),
            "metric_pair_fraction": len(values) / max(1, len(images)),
            "failure_counts": dict(reasons),
            "failure_examples": examples,
        }
        if values:
            report["markers"]["center_separation_p05_p50_p95_m"] = np.quantile(
                values, [0.05, 0.5, 0.95]
            ).tolist()
            report["markers"]["reprojection_p50_p95_px"] = np.quantile(
                errors, [0.5, 0.95]
            ).tolist()
    return report
