from pathlib import Path

import numpy as np
import pytest

from yumi.robot import KarmaModel, plan_targets, solve_ik


@pytest.fixture
def model():
    pytest.importorskip("mujoco")
    if not Path("research/i2rt/i2rt/robot_models/arm/yam/yam.xml").exists():
        pytest.skip("Run the pinned model setup in docs/single-arm-umi.md")
    return KarmaModel(
        "research/karma", "research/i2rt/i2rt/robot_models/arm/yam/yam.xml"
    )


def test_fk_ik_residuals_and_unreachable(model):
    rng = np.random.default_rng(4)
    for _ in range(12):
        target_q = np.array([0, 0.6, 0.6, 0, 0.3, 0]) + rng.uniform(-0.1, 0.1, 6)
        target = model.fk(target_q)
        _, error = solve_ik(model, target, target_q + rng.uniform(-0.03, 0.03, 6))
        assert error["position_error_m"] <= 1e-5
        assert error["rotation_error_rad"] <= 1e-4
    target[:3, 3] = [10, 10, 10]
    with pytest.raises(ValueError, match="IK not reached"):
        solve_ik(model, target, target_q)


def test_trajectory_limits_and_collision(model):
    q = np.array([0, 0.6, 0.6, 0, 0.3, 0])
    assert not model.penetrations(q)
    targets = [model.fk(q), model.fk(q + np.array([0.01, 0, 0, 0, 0, 0]))]
    plan = plan_targets(model, targets, [0, 0.1], q)
    assert len(plan["joint_positions_rad"]) == 2 and not plan["hardware_validated"]
    with pytest.raises(ValueError, match="speed limit"):
        plan_targets(model, targets, [0, 0.0001], q)
    with pytest.raises(ValueError, match="workspace"):
        plan_targets(model, targets, [0, 0.1], q, workspace=[[5, 5, 5], [6, 6, 6]])
    with pytest.raises(ValueError, match="motion limits"):
        plan_targets(model, targets, [0, 0.1], q, max_joint_speed=float("nan"))


def test_slow_smooth_motion_does_not_become_ik_stair_steps(model):
    q = np.array([0, 0.6, 0.6, 0, 0.3, 0])
    anchor = model.fk(q)
    targets = []
    for i in range(17):
        t = anchor.copy()
        t[:3, 3] += anchor[:3, 0] * i * 0.01 / 30
        targets.append(t)
    plan = plan_targets(
        model, targets, np.arange(17) / 30, q, max_joint_speed=0.5, max_joint_accel=2.0
    )
    assert len(plan["joint_positions_rad"]) == 17
