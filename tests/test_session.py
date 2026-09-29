import io
import threading
from types import SimpleNamespace

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


def test_ctrl_c_saves_short_episode_then_returns_to_prompt(tmp_path, monkeypatch):
    import yumi.session as module

    s = Session(tmp_path / "dataset", 3, 30, 0, "test", "local/test")
    s.meta = {"config": {"markers": {}}, "rgb_intrinsics": {}}
    monkeypatch.setattr("yumi.markers.Markers", lambda *a: object())
    monkeypatch.setattr(
        "yumi.capture_quality.CaptureQuality.update",
        lambda *a, **kw: {"ready": True, "width_m": 0.02, "reasons": []},
    )
    now = [10.0]
    latest = {}

    def clock():
        now[0] += 0.1
        latest.update(
            pose={"arrival_s": now[0], "tracker_confidence": 3}, rgb_arrival_s=now[0]
        )
        return now[0]

    monkeypatch.setattr(module.time, "monotonic", clock)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module.sys, "stdin", io.StringIO("\nq\n"))
    polls = [0]

    def poll(*args):
        polls[0] += 1
        return ([module.sys.stdin], [], []) if polls[0] > 15 else ([], [], [])

    monkeypatch.setattr(module.select, "select", poll)
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3)
    )

    def check():
        if s.active:
            raise KeyboardInterrupt

    s.check = check
    s.wait_reply = lambda event, tick: (
        {"frames": 4, "low_confidence_frames": 0} if event == "saved" else {}
    )
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
