import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from yumi.capture_quality import CaptureQuality
from yumi.geometry import pose_matrix
from yumi.umi import (
    decode_pose,
    encode_pose,
    make_chunks,
    robot_targets,
    width_to_command,
)


def test_pose_roundtrip_and_bad_rotations():
    a = pose_matrix(
        [0.3, -0.2, 0.1], Rotation.from_euler("xyz", [0.2, -0.4, 1.2]).as_quat()
    )
    decoded, width = decode_pose(encode_pose(a, 0.042))
    np.testing.assert_allclose(decoded, a, atol=1e-12)
    assert width == 0.042
    with pytest.raises(ValueError):
        decode_pose(np.zeros(10))


def test_robot_axis_conversion_is_conjugation_not_translation_only():
    anchor = pose_matrix([0.5, 0.1, 0.3], Rotation.from_euler("z", 0.7).as_quat())
    x = pose_matrix([0.01, 0.02, 0.03], Rotation.from_euler("x", 0.4).as_quat())
    delta = pose_matrix([0.02, -0.01, 0.04], Rotation.from_euler("y", 0.2).as_quat())
    out = robot_targets(anchor, [encode_pose(delta, 0.025)], x)[0][0]
    np.testing.assert_allclose(
        anchor @ np.linalg.inv(x) @ delta, out @ np.linalg.inv(x), atol=1e-12
    )


def test_chunks_do_not_bridge_invalid_frames_and_all_share_current_anchor():
    frames = [
        {
            "frame_index": i,
            "time_s": i / 30,
            "valid": True,
            "T_world_tcp": pose_matrix([i * 0.001, 0, 0], [0, 0, 0, 1]).tolist(),
            "opening_m": 0.02,
        }
        for i in range(12)
    ]
    frames[5]["valid"] = False
    chunks = make_chunks(frames, 2, 2, 30)
    for chunk in chunks:
        assert 5 not in chunk["history_frames"] + chunk["future_frames"]
        assert chunk["observation_state"][-1][0] == pytest.approx(0)
        assert chunk["observation_state"][0][0] == pytest.approx(-0.001)
        assert chunk["action"][0][0] == pytest.approx(0.001)
        assert chunk["action"][1][0] == pytest.approx(0.002)


def test_gripper_units_and_no_extrapolation():
    assert width_to_command(0.025, [0, 0.05], [0, 1]) == 0.5
    assert width_to_command(0.01, [0, 0.05], [0, 1], "dataset_closed") == 0.8
    with pytest.raises(ValueError):
        width_to_command(0.06, [0, 0.05], [0, 1])


def test_capture_cannot_start_on_tracking_alone():
    class Detector:
        def width(self, _):
            raise ValueError("Missing or duplicate marker 13")

    gate = CaptureQuality(Detector())
    status = gate.update(
        {
            "rgb": np.zeros((2, 2, 3)),
            "rgb_arrival_s": 10,
            "pose": {"arrival_s": 10, "tracker_confidence": 3},
        },
        now=10,
    )
    assert not status["ready"]
    assert "Missing or duplicate marker 13" in status["reasons"]
