"""Generate clearly labelled synthetic raw captures to exercise the entire pipeline."""

import json
from pathlib import Path

import cv2
import numpy as np

from yumi.processing import prepare


def generate(root, episode):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "rgb").mkdir()
    (root / "depth").mkdir()
    config = json.loads(Path("configs/umi.example.json").read_text())
    config.update(
        tracking_serial="SYNTHETIC",
        rgb_serial="SYNTHETIC",
        calibration_verified=True,
        timing_verified=True,
        T_tracker_tcp=np.eye(4).tolist(),
        warmup_s=0.1,
        task="SYNTHETIC SOFTWARE TEST — not a real demonstration",
    )
    config["markers"].update(size_m=0.02, width_gain=1.0, width_offset_m=-0.02)
    intr = {
        "width": 640,
        "height": 480,
        "fx": 600.0,
        "fy": 600.0,
        "ppx": 320.0,
        "ppy": 240.0,
        "coeffs": [0.0] * 5,
        "model": "distortion.none",
    }
    meta = {"config": config, "rgb_intrinsics": intr, "synthetic": True}
    (root / "metadata.json").write_text(json.dumps(meta))
    rows = []
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i in range(45):
        rgb = np.full((480, 640, 3), 255, np.uint8)
        for marker, x in [(13, 240), (14, 360)]:
            rgb[220:260, x : x + 40] = cv2.aruco.generateImageMarker(
                dictionary, marker, 40
            )[:, :, None]
        cv2.putText(
            rgb,
            f"SYNTHETIC episode {episode} frame {i}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.imwrite(str(root / f"rgb/{i:08d}.png"), rgb)
        np.save(root / f"depth/{i:08d}.npy", np.full((480, 640), 300, np.uint16))
        rows.append(
            {
                "kind": "image",
                "index": i,
                "device_s": 100 + i / 30,
                "arrival_s": 10000 + i / 30,
                "frame_number": i,
                "domain": "hardware_clock",
                "rgb_path": f"rgb/{i:08d}.png",
                "depth_path": f"depth/{i:08d}.npy",
            }
        )
    for i in range(300):
        rows.append(
            {
                "kind": "pose",
                "device_s": 200 + i / 200,
                "arrival_s": 10000 + i / 200,
                "frame_number": i,
                "domain": "hardware_clock",
                "position": [i / 200 * 0.01, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
                "tracker_confidence": 3,
                "mapper_confidence": 3,
            }
        )
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "COMPLETE").touch()
    result = prepare(root, root / "prepared.json")
    print(result)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default="artifacts/synthetic")
    a = p.parse_args()
    for i in range(2):
        generate(Path(a.root) / f"episode_{i}", i)
