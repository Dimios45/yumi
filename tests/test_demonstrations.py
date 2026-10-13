import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from yumi.demonstrations import (
    bounded_widths,
    complete_episode_segment,
    prepare_episode,
    validate_stream,
)


def test_width_interpolation_never_extrapolates_or_fills_long_gaps():
    times = np.arange(7) / 30
    widths = [np.nan, 0.02, np.nan, 0.04, np.nan, np.nan, np.nan]
    filled, measured = bounded_widths(times, widths, 0.07)
    assert filled[2] == pytest.approx(0.03)
    assert np.isnan(filled[[0, 4, 5, 6]]).all()
    assert not measured[2]
    strict, _ = bounded_widths(times, widths)
    assert np.isnan(strict[2])
    with pytest.raises(ValueError):
        bounded_widths(times, widths, float("nan"))


def test_stream_resets_and_domain_changes_are_rejected():
    rows = [
        {"domain": "global", "device_s": i / 200, "frame_number": i} for i in range(3)
    ]
    assert len(validate_stream(rows, "pose")) == 3
    rows[2]["frame_number"] = 0
    with pytest.raises(ValueError, match="reset"):
        validate_stream(rows, "pose")
    rows[2]["domain"] = "hardware"
    with pytest.raises(ValueError, match="domain"):
        validate_stream(rows, "pose")


def test_short_surviving_fragment_is_not_a_complete_demonstration():
    doc = {
        "report": {"images": 150},
        "segments": [{"frames": [{"source_frame": i} for i in range(3, 148)]}],
    }
    assert len(complete_episode_segment(doc)["frames"]) == 145
    doc["segments"][0]["frames"] = [{"source_frame": i} for i in range(25, 148)]
    with pytest.raises(ValueError, match="omits too much"):
        complete_episode_segment(doc)
    doc["segments"].append({"frames": []})
    with pytest.raises(ValueError, match="one clean segment"):
        complete_episode_segment(doc)


def test_high_confidence_jump_between_rgb_frames_splits_episode(tmp_path, monkeypatch):
    config = json.loads(
        (Path(__file__).parents[1] / "configs/umi.example.json").read_text()
    )
    config.update(
        T_tracker_tcp=np.eye(4).tolist(), camera_time_offset_s=0, resolution=[32, 32]
    )

    class Detector:
        def __init__(self, *args):
            pass

        def width(self, rgb):
            return 0.03, 0, []

    monkeypatch.setattr("yumi.demonstrations.Markers", Detector)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"config": config, "rgb_intrinsics": {}})
    )
    (tmp_path / "COMPLETE").touch()
    cv2.imwrite(str(tmp_path / "rgb.png"), np.zeros((32, 32, 3), np.uint8))
    rows = []
    for i in range(300):
        rows.append(
            {
                "kind": "pose",
                "device_s": 1000 + i / 200,
                "arrival_s": 2000 + i / 200,
                "frame_number": i,
                "domain": "timestamp_domain.global_time",
                "position": [0.05 if i == 121 else 0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
                "tracker_confidence": 3,
            }
        )
    for i in range(40):
        rows.append(
            {
                "kind": "image",
                "device_s": 1000.001 + i / 30,
                "arrival_s": 2000.001 + i / 30,
                "frame_number": i,
                "domain": "timestamp_domain.global_time",
                "rgb_path": "rgb.png",
            }
        )
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    doc = prepare_episode(
        tmp_path, allow_provisional=True, require_depth=False, min_seconds=0.2
    )
    assert doc["report"]["raw_bad_intervals"] == 2
    assert len(doc["segments"]) == 2
    with pytest.raises(ValueError, match="one clean segment"):
        complete_episode_segment(doc)
