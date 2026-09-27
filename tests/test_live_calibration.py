import json

import pytest

from yumi.live_calibration import StablePair, append_sample


def ready_pair():
    pair = StablePair()
    for i in range(5):
        pair.update(10 + i * 0.03, i, 100 + i * 0.1)
    return pair


def test_stable_pair_expires_and_refuses_overlapping_resave():
    pair = ready_pair()
    snapshot = pair.snapshot(10.2)
    assert snapshot["valid_frames"] == 5
    pair.last_used = snapshot["frame_number"]
    with pytest.raises(ValueError, match="already saved"):
        pair.snapshot(10.2)
    for i in range(5, 10):
        pair.update(10 + i * 0.03, i, 100.0)
    assert pair.snapshot(10.3)["frame_number"] == 9
    with pytest.raises(ValueError, match="fresh"):
        pair.snapshot(13.0)


def test_missing_frames_break_stable_window_but_short_label_window_survives():
    pair = ready_pair()
    pair.update(10.15, 5, reason="Missing marker 13")
    assert pair.snapshot(10.2)["frame_number"] == 4
    for i in range(6, 10):
        pair.update(10 + i * 0.03, i, 120.0)
    assert pair.ready["frame_number"] == 4
    pair.update(10.3, 10, 120.0)
    assert pair.snapshot(10.4)["marker_separation_px"] == 120.0


def test_motion_clears_old_ready_sample():
    pair = ready_pair()
    pair.update(10.15, 5, 120.0)
    with pytest.raises(ValueError, match="fresh"):
        pair.snapshot(10.2)


def test_append_is_compatible_with_fitter_and_checks_identity(tmp_path):
    config = {
        "rgb_serial": "test",
        "resolution": [640, 480],
        "markers": {
            "dictionary": "DICT_4X4_50",
            "ids": [13, 14],
            "size_m": 0.01,
            "width_range_m": [0, 0.1],
        },
    }
    path = tmp_path / "width.json"
    result = append_sample(path, config, ready_pair().snapshot(10.2), 30.0)
    assert result["sample_count"] == 1
    doc = json.loads(path.read_text())
    assert doc["jaw_opening_m"] == [0.03]
    assert doc["marker_separation_px"] == [100.2]
    config["rgb_serial"] = "other"
    with pytest.raises(ValueError, match="match"):
        append_sample(path, config, ready_pair().snapshot(10.2), 30.0)
    assert len(json.loads(path.read_text())["captures"]) == 1
