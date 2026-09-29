"""Reproducible, offline-only feasibility plan for one prepared UMI action chunk."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .robot import KarmaModel, plan_targets
from .umi import CONTRACT, robot_targets, width_to_command

KARMA_REV = "b4f06f6d645755e605b6c0aec7c10af3d2c911d6"
I2RT_REV = "5d47b358bafb30c65e397f2ece506550a0db4594"


def revision(root, expected):
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if actual != expected or dirty:
        raise ValueError(f"Model checkout must be clean at {expected}: {root}")
    return actual


def make_plan(manifest, profile, karma_root, i2rt_root, sample_index=0):
    doc = json.loads(Path(manifest).read_text())
    cfg = json.loads(Path(profile).read_text())
    workspace = np.asarray(cfg["workspace_m"], float)
    if (
        workspace.shape != (2, 3)
        or not np.isfinite(workspace).all()
        or np.any(workspace[0] >= workspace[1])
    ):
        raise ValueError(
            "Fill workspace_m with finite [min_xyz, max_xyz] robot-base bounds"
        )
    if doc["contract"] != CONTRACT:
        raise ValueError("Unsupported action contract")
    if not 0 <= sample_index < len(doc["samples"]):
        raise ValueError("No such valid action chunk")
    revisions = {
        "karma": revision(karma_root, KARMA_REV),
        "i2rt": revision(i2rt_root, I2RT_REV),
    }
    model = KarmaModel(
        karma_root, Path(i2rt_root) / "i2rt/robot_models/arm/yam/yam.xml"
    )
    seed = np.asarray(cfg["initial_joints_rad"], float)
    anchor = model.fk(seed)
    sample = doc["samples"][sample_index]
    pairs = robot_targets(anchor, sample["action"], cfg["T_capture_tcp_robot_tool"])
    plan = plan_targets(
        model,
        [anchor] + [t for t, _ in pairs],
        [0] + sample["action_time_offsets_s"],
        seed,
        cfg["max_joint_speed_rad_s"],
        cfg["max_joint_accel_rad_s2"],
        workspace,
    )
    measured = cfg["gripper"]
    widths = [sample["observation_state"][-1][-1]] + [w for _, w in pairs]
    commands = [
        width_to_command(w, measured["widths_m"], measured["native_open_commands"])
        for w in widths
    ]
    plan.update(
        contract=CONTRACT,
        source_manifest=str(Path(manifest).resolve()),
        sample_index=sample_index,
        source_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        profile=cfg,
        model_revisions=revisions,
        gripper_native_open=commands,
        calibration_verified=doc["calibration_verified"],
        hardware_execution_allowed=False,
        note="Offline feasibility only. This file is not an execution authorization or a trained policy.",
    )
    return plan
