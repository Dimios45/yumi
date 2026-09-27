import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from yumi.calibration import fit_width, hand_eye
from yumi.geometry import (
    check_motion,
    fit_clock,
    interpolate_pose,
    pose_matrix,
    transform,
)


def test_transform_rejects_reflection():
    a = np.eye(4)
    a[0, 0] = -1
    with pytest.raises(ValueError):
        transform(a)


def test_interpolation_shortest_quaternion_and_no_extrapolation():
    q = np.array([[0, 0, 0, 1], [0, 0, 0, -1.0]])
    a = interpolate_pose([0, 0.01], np.array([[0, 0, 0], [1, 0, 0]]), q, [3, 3], 0.005)
    np.testing.assert_allclose(a[:3, 3], [0.5, 0, 0])
    np.testing.assert_allclose(a[:3, :3], np.eye(3))
    for t in [-0.001, 0.011]:
        with pytest.raises(ValueError):
            interpolate_pose([0, 0.01], np.zeros((2, 3)), q, [3, 3], t)
    with pytest.raises(ValueError):
        interpolate_pose([0, 0.01], np.zeros((2, 3)), q, [3, 1], 0.005)


def test_clock_independent_epochs_and_drift():
    x = 4000 + np.arange(1000) * 0.005
    y = 10000 + 1.0001 * (x - x[0])
    fitted, report = fit_clock(x, y)
    np.testing.assert_allclose(fitted, y, atol=1e-8)
    assert report["rate"] == pytest.approx(1.0001)
    with pytest.raises(ValueError):
        fit_clock([0] * 10, range(10))


def test_rotated_tcp_offset():
    a = pose_matrix([1, 0, 0], Rotation.from_euler("z", 90, degrees=True).as_quat())
    ext = np.eye(4)
    ext[0, 3] = 0.2
    np.testing.assert_allclose((a @ ext)[:3, 3], [1, 0.2, 0], atol=1e-8)


def test_jump_rejection():
    a = np.repeat(np.eye(4)[None], 3, axis=0)
    a[-1, 0, 3] = 1
    with pytest.raises(ValueError):
        check_motion([0, 0.01, 0.02], a, 2, 10)


def test_width_calibration():
    x = np.linspace(0.02, 0.12, 12)
    r = fit_width(x, x - 0.02)
    assert r["width_gain"] == pytest.approx(1)
    assert r["width_offset_m"] == pytest.approx(-0.02)


def test_hand_eye_recovers_transform():
    rng = np.random.default_rng(7)
    x = pose_matrix(
        [0.04, 0.01, -0.07], Rotation.from_euler("xyz", [0.2, -0.4, 0.1]).as_quat()
    )
    target = pose_matrix([0.2, 0.1, 0.8], [0, 0, 0, 1])
    aa = []
    bb = []
    for _ in range(20):
        a = pose_matrix(
            rng.normal(0, 0.1, 3), Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_quat()
        )
        aa.append(a)
        bb.append(np.linalg.inv(x) @ np.linalg.inv(a) @ target)
    r = hand_eye(aa, bb)
    np.testing.assert_allclose(r["T_tracker_camera"], x, atol=1e-8)
    assert r["board_position_rms_m"] < 1e-8


def test_temporal_calibration_sign():
    from yumi.calibration import estimate_time_offset

    def angle(t):
        return 0.7 * np.sin(1.3 * t) + 0.3 * np.sin(3.7 * t) + 0.2 * np.sin(6.1 * t)

    tt = np.arange(0, 8, 0.005)
    ct = np.arange(0.3, 7.7, 1 / 30)
    delay = 0.045
    cr = Rotation.from_euler("z", -angle(ct + delay)).as_matrix()
    tr = Rotation.from_euler("z", angle(tt)).as_matrix()
    result = estimate_time_offset(ct, cr, tt, tr)
    assert result["camera_time_offset_s"] == pytest.approx(delay, abs=0.003)


def test_global_clocks_keep_acquisition_offset_despite_delivery_latency():
    from yumi.geometry import synchronize_streams

    image = [
        {
            "device_s": 100 + i * 0.03,
            "arrival_s": 500 + i * 0.03 + 0.06,
            "domain": "timestamp_domain.global_time",
        }
        for i in range(30)
    ]
    pose = [
        {
            "device_s": 100.012 + i * 0.005,
            "arrival_s": 500.012 + i * 0.005 + 0.001,
            "domain": "timestamp_domain.global_time",
        }
        for i in range(100)
    ]
    ti, tp, a, b = synchronize_streams(image, pose)
    assert tp[0] - ti[0] == pytest.approx(0.012)
    assert a["alignment"] == b["alignment"] == "shared_sdk_global_time"


def test_mixed_clock_domain_is_rejected():
    from yumi.geometry import synchronize_streams

    a = [
        {
            "device_s": i * 0.01,
            "arrival_s": i * 0.01 + 2,
            "domain": "timestamp_domain.global_time",
        }
        for i in range(30)
    ]
    b = [dict(r, domain="timestamp_domain.hardware_clock") for r in a]
    with pytest.raises(ValueError, match="Mixed"):
        synchronize_streams(a, b)
