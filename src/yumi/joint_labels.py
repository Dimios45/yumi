"""Offline IK-derived six-arm-joint + metric aperture labels. No motor access."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .demonstrations import CONTRACT, write_manifest
from .geometry import transform
from .robot import KarmaModel, plan_targets
from .umi import encode_pose


def model_provenance(model):
    paths = [model.arm_xml, model.karma_root / "src/vr_teleop_kit/ik/model.py"]
    # Include the gripper XML and all mesh assets, not only a source revision.
    robot_models = model.arm_xml.parents[2]
    paths += sorted((robot_models / "gripper/linear_4310").rglob("*.xml"))
    paths += sorted(model.arm_xml.parent.rglob("*.stl"))
    paths += sorted((robot_models / "gripper/linear_4310").rglob("*.stl"))
    result = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    revisions = {}
    for label, path in (("karma", model.karma_root), ("i2rt", model.arm_xml.parent)):
        revisions[label] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        revisions[label + "_dirty"] = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(path),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
            ).strip()
        )
    return {"files_sha256": result, "revisions": revisions}


def derive_joints(doc, model, profile, *, allow_provisional=False):
    if doc["contract"] != CONTRACT:
        raise ValueError("Unsupported demonstration contract")
    registered = bool(profile.get("registration_verified"))
    if (not doc["calibration_verified"] or not registered) and not allow_provisional:
        raise ValueError(
            "Unverified capture/robot registration requires --allow-provisional"
        )
    seed = np.asarray(profile["initial_joints_rad"], float)
    if seed.shape != (6,):
        raise ValueError("Robot profile needs six initial arm joints")
    x = transform(profile["T_capture_tcp_robot_tool"])
    anchor = model.fk(seed)
    ix = np.linalg.inv(x)
    retained, rejected = [], []
    for segment in doc["segments"]:
        frames = segment["frames"]
        initial = np.linalg.inv(transform(frames[0]["T_world_tcp"]))
        targets = [
            transform(anchor @ ix @ initial @ transform(f["T_world_tcp"]) @ x)
            for f in frames
        ]
        try:
            plan = plan_targets(
                model,
                targets,
                [f["time_s"] for f in frames],
                seed,
                profile["max_joint_speed_rad_s"],
                profile["max_joint_accel_rad_s2"],
                profile.get("workspace_m"),
            )
        except ValueError as e:
            rejected.append(
                {"first_source_frame": frames[0]["source_frame"], "reason": str(e)}
            )
            continue
        for f, target, q, residual in zip(
            frames, targets, plan["joint_positions_rad"], plan["residuals"]
        ):
            f["joint_state"] = [*q, f["opening_m"]]
            f["ee_pose_robot"] = encode_pose(target, f["opening_m"]).tolist()
            f["ik_residual"] = residual
        retained.append(segment | {"ik_report": plan})
    return doc | {
        "segments": retained,
        "joint_label_source": "IK-derived; not measured robot feedback",
        "robot_profile": profile,
        "robot_registration_verified": registered,
        "robot_model": model_provenance(model),
        "joint_rejected_segments": rejected,
        "hardware_validated": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--karma-root", required=True)
    p.add_argument("--arm-xml", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-provisional", action="store_true")
    a = p.parse_args()
    doc = derive_joints(
        json.loads(Path(a.manifest).read_text()),
        KarmaModel(a.karma_root, a.arm_xml),
        json.loads(Path(a.profile).read_text()),
        allow_provisional=a.allow_provisional,
    )
    write_manifest(doc, a.output)
    print(
        json.dumps(
            {
                "segments": len(doc["segments"]),
                "rejected": doc["joint_rejected_segments"],
            }
        )
    )
    return 0 if doc["segments"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
