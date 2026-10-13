"""Read-only sensor-log audit. No cameras, robot connections, or label repair.

Run with the capture environment; output is diagnostic, not an acceptance gate.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from yumi.geometry import synchronize_streams


def audit(root):
    metadata_path = root / "session.json"
    meta = json.loads(metadata_path.read_text())
    config = meta["config"]
    episodes = []
    for source in sorted((root / "extras").glob("episode_*/samples.jsonl")):
        payload = source.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        images = [r for r in rows if r["kind"] == "image"]
        poses = [r for r in rows if r["kind"] == "pose"]
        widths = [
            r["width_measurement"]["opening_m"]
            for r in images
            if r.get("width_measurement", {}).get("valid")
        ]
        result = {
            "episode": source.parent.name,
            "samples_sha256": hashlib.sha256(payload).hexdigest(),
            "stream_counts": dict(Counter(r["kind"] for r in rows)),
            "width_valid_frames": len(widths),
            "width_invalid_reasons": dict(
                Counter(
                    r.get("width_error", "unspecified")
                    for r in images
                    if not r.get("width_measurement", {}).get("valid")
                )
            ),
            "width_range_mm": [float(min(widths) * 1000), float(max(widths) * 1000)]
            if widths
            else None,
            "confidence_counts": dict(Counter(p["tracker_confidence"] for p in poses)),
        }
        for label, stream in (("rgb", images), ("pose", poses)):
            delta = np.diff([r["device_s"] for r in stream])
            counters = np.diff([r["frame_number"] for r in stream])
            result[label + "_timing"] = {
                "nonincreasing_timestamps": int(np.sum(delta <= 0)),
                "frame_counter_gaps": int(np.sum(counters > 1)),
                "missing_frame_numbers": int(np.maximum(counters - 1, 0).sum()),
                "nonincreasing_frame_numbers": int(np.sum(counters <= 0)),
                "max_gap_ms": float(delta.max() * 1000) if len(delta) else None,
                "median_gap_ms": float(np.median(delta) * 1000) if len(delta) else None,
            }
        try:
            ti, tp, image_clock, pose_clock = synchronize_streams(images, poses)
            ti = ti + config["camera_time_offset_s"]
            j = np.searchsorted(tp, ti, side="right")
            inside = (j > 0) & (j < len(tp))
            right = np.clip(j, 1, len(tp) - 1)
            confidence = np.asarray([p["tracker_confidence"] for p in poses])
            good = (
                inside
                & (tp[right] - tp[right - 1] <= config["max_pose_gap_s"])
                & (confidence[right] >= config["min_tracker_confidence"])
                & (confidence[right - 1] >= config["min_tracker_confidence"])
            )
            width_valid = np.asarray(
                [r.get("width_measurement", {}).get("valid", False) for r in images]
            )
            result.update(
                pose_bracket_valid_frames=int(good.sum()),
                pose_and_width_valid_frames=int((good & width_valid).sum()),
                clocks={"rgb": image_clock, "pose": pose_clock},
            )
        except (ValueError, IndexError) as error:
            result["synchronization_error"] = str(error)
        episodes.append(result)
    return {
        "source": str(root.resolve()),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "complete": (root / "COMPLETE").exists(),
        "failed": (root / "FAILED.txt").exists(),
        "state_pose_frame": meta.get("state_pose_frame"),
        "has_tracker_tcp": config.get("T_tracker_tcp") is not None,
        "calibration_verified": config["calibration_verified"],
        "timing_verified": config["timing_verified"],
        "limitations": "Sensor logs only: no video/depth decoding, physical accuracy, discontinuity, collision, IK or policy acceptance. Pose counts check timestamp bracketing and confidence only.",
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.session)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    for episode in report["episodes"]:
        print(
            episode["episode"],
            "images",
            episode["stream_counts"].get("image", 0),
            "valid widths",
            episode["width_valid_frames"],
            "valid pose brackets + widths",
            episode.get("pose_and_width_valid_frames", "unknown"),
        )


if __name__ == "__main__":
    main()
