"""Synthetic session -> synchronized UMI manifests. No camera or robot needed."""

import argparse
import json
from pathlib import Path

import cv2
from synthetic_smoke import generate

from yumi.session_processing import prepare_session


def run(root):
    root = Path(root)
    generate(root / "legacy", 0)
    source = root / "legacy"
    session = root / "session"
    extras = session / "extras/episode_000000"
    extras.mkdir(parents=True)
    (session / "meta").mkdir()
    video = session / "videos/observation.images.d405/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
    )
    for file in sorted((source / "rgb").glob("*.png")):
        writer.write(cv2.imread(str(file)))
    writer.release()
    (extras / "samples.jsonl").write_bytes((source / "samples.jsonl").read_bytes())
    (session / "session.json").write_bytes((source / "metadata.json").read_bytes())
    (session / "meta/info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_frames": 45})
    )
    (session / "COMPLETE").touch()
    report = prepare_session(session, root / "prepared")
    assert report["training_ready"] and report["total_chunks"] == 28, report
    print(json.dumps(report, indent=2))
    # A short tracking failure can fall strictly between two RGB timestamps.
    # It must still break the action window, even when both RGB brackets look good.
    log = extras / "samples.jsonl"
    original = log.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    for row in rows:
        if row["kind"] == "pose" and row["frame_number"] == 142:
            row["tracker_confidence"] = 0
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    try:
        rejected = prepare_session(session, root / "tracking-loss-test")
        assert 0 < rejected["total_chunks"] < report["total_chunks"]
        assert (
            rejected["episodes"][0]["invalid_reasons"][
                "Tracking loss between image frames"
            ]
            >= 2
        )
    finally:
        log.write_text(original)
    print("Between-frame tracking-loss rejection passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
