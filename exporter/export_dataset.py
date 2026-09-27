"""Run ONLY in exporter uv project: official LeRobot v3 writer and readback."""

import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

NAMES = ["x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "aperture_m"]
IMAGE = "observation.images.d405"


def verify(root, repo_id):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Not LeRobot v3")
    ds = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    if len(ds) != info["total_frames"] or ds.num_episodes != info["total_episodes"]:
        raise ValueError("Metadata count mismatch")
    if len(ds) == 0:
        raise ValueError("Empty dataset")
    previous = None
    for i in range(len(ds)):
        row = ds[i]  # Official decoder follows v3 per-episode video offsets.
        image = row[IMAGE]
        if image.ndim != 3 or image.shape[0] != 3 or not image.isfinite().all():
            raise ValueError("Invalid decoded RGB")
        s, a = row["observation.state"].numpy(), row["action"].numpy()
        if not np.isfinite([s, a]).all():
            raise ValueError("Nonfinite state/action")
        if not np.allclose(
            [np.linalg.norm(s[3:7]), np.linalg.norm(a[3:7])], 1, atol=1e-5
        ):
            raise ValueError("Quaternion norm")
        episode = int(row["episode_index"])
        if (
            previous is not None
            and previous[0] == episode
            and not np.allclose(previous[1], s, atol=1e-6)
        ):
            raise ValueError("Next-frame action mismatch")
        previous = episode, a
    decoded_frames = 0
    for path in (root / "videos").rglob("*.mp4"):
        with av.open(str(path)) as container:
            decoded_frames += sum(1 for _ in container.decode(video=0))
    if decoded_frames != len(ds):
        raise ValueError(f"Video frames {decoded_frames} != parquet frames {len(ds)}")
    report = {
        "format": info["codebase_version"],
        "episodes": ds.num_episodes,
        "frames": len(ds),
        "decoded_video_frames": decoded_frames,
        "official_loader_readback": "all frames passed",
    }
    (root / "verification.json").write_text(json.dumps(report, indent=2))
    return report


def export(manifests, root, repo_id):
    root = Path(root)
    staging = root.with_name(root.name + ".partial")
    if root.exists() or staging.exists():
        raise FileExistsError("Use a new output directory; no dataset is overwritten")
    docs = [json.loads(Path(p).read_text()) for p in manifests]
    if not docs or not docs[0]["samples"]:
        raise ValueError("No prepared samples")
    first_image = Path(docs[0]["raw_root"]) / docs[0]["samples"][0]["rgb_path"]
    with Image.open(first_image) as im:
        w, h = im.size
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": NAMES},
        IMAGE: {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.capture_time": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["seconds_from_episode_origin"],
        },
        "observation.quality": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["image_skew_s", "marker_reprojection_px"],
        },
    }
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        root=staging,
        fps=docs[0]["fps"],
        robot_type="custom_umi_t265_d405",
        features=features,
        image_writer_threads=4,
        video_backend="pyav",
        vcodec="h264",
    )
    try:
        for episode, doc in enumerate(docs):
            if doc["fps"] != docs[0]["fps"]:
                raise ValueError("Episode FPS mismatch")
            raw = Path(doc["raw_root"])
            extras = staging / "extras" / f"episode_{episode:06d}"
            extras.mkdir(parents=True)
            # Preserve metric Z16 depth and full-rate pose/IMU with provenance.
            # These extras are not lossy video or automatic LeRobot policy inputs.
            shutil.copytree(raw / "depth", extras / "depth")
            for name in ("metadata.json", "samples.jsonl"):
                shutil.copy2(raw / name, extras / name)
            (extras / "prepared.json").write_text(json.dumps(doc, indent=2))
            for sample in doc["samples"]:
                with Image.open(raw / sample["rgb_path"]) as image:
                    rgb = np.asarray(image.convert("RGB"))
                if rgb.shape != (h, w, 3):
                    raise ValueError("RGB resolution mismatch")
                ds.add_frame(
                    {
                        "observation.state": np.asarray(sample["state"], np.float32),
                        "action": np.asarray(sample["action"], np.float32),
                        IMAGE: rgb,
                        "observation.capture_time": np.asarray(
                            [sample["sensor_time_s"]], np.float64
                        ),
                        "observation.quality": np.asarray(
                            [sample["image_skew_s"], sample["marker_error_px"]],
                            np.float32,
                        ),
                        "task": doc["task"],
                    }
                )
            ds.save_episode()
    finally:
        ds.finalize()
    report = verify(staging, repo_id)
    staging.rename(root)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["export", "verify"])
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default="local/custom-umi")
    p.add_argument("--manifests", nargs="+")
    a = p.parse_args()
    if a.command == "export" and not a.manifests:
        p.error("--manifests is required for export")
    print(
        json.dumps(
            export(a.manifests, a.root, a.repo_id)
            if a.command == "export"
            else verify(a.root, a.repo_id),
            indent=2,
        )
    )
