import io
import json
import pickle
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yumi.session import Session


def test_idle_samples_are_not_written_and_episode_boundaries_are_ordered(tmp_path):
    s = Session(tmp_path / "dataset", 2, 30, 30, "test", "local/test")
    s.accept({"kind": "pose", "frame_number": 0})
    assert s.q.empty()
    s.begin()
    s.accept({"kind": "pose", "frame_number": 1})
    s.end()
    s.accept({"kind": "pose", "frame_number": 2})
    assert [
        s.q.get()["command"],
        s.q.get()["sample"]["frame_number"],
        s.q.get()["command"],
    ] == ["begin", 1, "end"]
    assert s.q.empty()


@pytest.mark.parametrize(
    "health",
    [
        {"ready": True, "width_m": 0.02, "reasons": []},
        # SPACE still starts while a marker is hidden; the problem is saved per frame.
        {
            "ready": False,
            "sensor_ready": False,
            "width_m": None,
            "reasons": ["Missing or duplicate marker 14"],
        },
    ],
)
def test_ctrl_c_saves_short_episode_then_returns_to_prompt(
    tmp_path, monkeypatch, health
):
    import yumi.session as module

    s = Session(tmp_path / "dataset", 3, 30, 0, "test", "local/test")
    s.meta = {"config": {"markers": {}}, "rgb_intrinsics": {}}
    monkeypatch.setattr("yumi.markers.Markers", lambda *a: object())
    monkeypatch.setattr(
        "yumi.capture_quality.CaptureQuality.update",
        lambda *a, **kw: health,
    )
    now = [10.0]
    latest = {}

    def clock():
        now[0] += 0.1
        if round(now[0], 1) == 12.0:  # SPACE once tracking has been ready for over 1 s
            s.keys.put("toggle")
        latest.update(
            pose={"arrival_s": now[0], "tracker_confidence": 3}, rgb_arrival_s=now[0]
        )
        return now[0]

    monkeypatch.setattr(module.time, "monotonic", clock)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    import contextlib

    @contextlib.contextmanager
    def keyboard(keys):  # SPACE is pressed in clock(); q after the save below
        yield

    monkeypatch.setattr("yumi.hardware._keyboard", keyboard)
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3)
    )

    def check():
        if s.active:
            raise KeyboardInterrupt

    s.check = check

    def wait_reply(event, tick):
        if event != "saved":
            return {}
        s.keys.put("quit")
        return {"frames": 4, "low_confidence_frames": 0}

    s.wait_reply = wait_reply
    s.run(latest, threading.Lock(), None, threading.Event())
    assert s.saved == 1 and not s.active
    assert s.q.get()["command"] == "begin"
    assert s.q.get()["command"] == "end"


def test_queue_overload_is_explicit_failure(tmp_path):
    s = Session(tmp_path / "dataset", 1, 30, 0, "test", "local/test")
    s.begin()
    for _ in range(511):
        s.accept({"kind": "pose"})
    with pytest.raises(RuntimeError, match="cannot keep up"):
        s.accept({"kind": "pose"})


def test_fresh_start_never_overwrites_existing_dataset(tmp_path):
    with pytest.raises(FileExistsError):
        Session(tmp_path, 1, 30, 30, "test", "local/test")


def test_raw_session_null_tcp_keeps_tracker_pose_and_flags_missing_width(tmp_path):
    s = Session(tmp_path / "raw", 3, 15, 0, "test", "local/test")
    config = json.loads(
        (Path(__file__).parents[1] / "configs/umi.example.json").read_text()
    )
    config["T_tracker_tcp"] = None
    config["markers"]["size_m"] = 0.01
    s.meta = {
        "config": config,
        "rgb_intrinsics": {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240},
    }
    pipe = io.BytesIO()
    s.process = SimpleNamespace(stdin=pipe)
    pose = {
        "kind": "pose",
        "position": [0.1, 0.2, 0.3],
        "quaternion_xyzw": [0, 0, 0, 1],
        "device_s": 1.0,
        "domain": "global",
        "tracker_confidence": 3,
    }
    image = {
        "kind": "image",
        "rgb": np.zeros((480, 640, 3), np.uint8),
        "device_s": 1.01,
        "domain": "global",
    }
    s.q.put({"sample": pose})
    s.q.put({"sample": image})
    s.q.put(None)
    s._write()
    assert s.failure is None
    pipe.seek(0)
    size = struct.unpack("!Q", pipe.read(8))[0]
    pickle.loads(pipe.read(size))
    size = struct.unpack("!Q", pipe.read(8))[0]
    actual = pickle.loads(pipe.read(size))["sample"]
    np.testing.assert_allclose(actual["provisional_state"][:3], [0.1, 0.2, 0.3])
    assert actual["quality"][1] == 1
    assert actual["quality"][2] == 0
    assert actual["width_measurement"] == {"valid": False, "opening_m": None}


def test_raw_readiness_does_not_require_calibrated_width():
    from yumi.capture_quality import CaptureQuality

    class InvalidWidth:
        def width(self, rgb):
            raise ValueError("Marker separation outside measured calibration range")

    quality = CaptureQuality(InvalidWidth())
    latest = {
        "rgb": np.zeros((2, 2, 3), np.uint8),
        "rgb_arrival_s": 10.0,
        "pose": {"arrival_s": 10.0, "tracker_confidence": 3},
    }
    health = quality.update(latest, now=10.1)
    assert health["sensor_ready"] and not health["ready"]
    assert health["width_m"] is None
    assert not quality.update(latest, now=11.0)["sensor_ready"]
