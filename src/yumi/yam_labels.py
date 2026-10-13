"""Offline labels using the installed i2rt YAM FK and Mink IK; no robot connection."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import interpolate_pose, pose_matrix, synchronize_streams, transform
from .markers import Markers
from .umi import encode_pose

CONTRACT = "yumi.i2rt_yam.normalized_gripper.v1"


def normalized_open(width, limits):
    lo, hi = map(float, limits)
    if not np.isfinite([lo, hi, width]).all() or hi <= lo or not lo <= width <= hi:
        raise ValueError("Opening outside measured UMI range")
    return (width - lo) / (hi - lo)


def model_setup():
    import mink
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    kin = Kinematics(xml, "grasp_site")
    model = kin._configuration.model
    if model.nq != 8:
        raise ValueError("Expected six arm joints and two linear gripper sliders")
    for i in range(6):
        if model.joint(i).name != f"joint{i+1}":
            raise ValueError("Unexpected i2rt joint ordering")
    limits = [mink.ConfigurationLimit(model)]
    return kin, model, limits, Path(xml)


def annotate(root, profile, output):
    import i2rt

    root, output = Path(root).resolve(), Path(output).resolve()
    if not (root / "COMPLETE").exists() or (root / "FAILED.txt").exists():
        raise ValueError("Raw session did not finish successfully")
    meta = json.loads((root / "session.json").read_text())
    c = meta["config"]
    ext = transform(c["T_tracker_tcp"])
    x = transform(profile["T_capture_tcp_robot_tool"])
    seed = np.asarray(profile["initial_joints_rad"], float)
    kin, model, limits, xml = model_setup()
    if seed.shape != (6,) or not np.isfinite(seed).all() or np.any(seed < model.jnt_range[:6, 0]) or np.any(seed > model.jnt_range[:6, 1]):
        raise ValueError("Invalid YAM initial arm state")
    anchor = kin.fk(np.r_[seed, np.zeros(2)]).copy()
    detector = Markers(c["markers"], meta["rgb_intrinsics"])
    native_limits = model.jnt_range[6:8].copy()
    i2rt_root = Path(i2rt.__file__).resolve().parents[1]
    sources = [i2rt_root / "i2rt/robots/kinematics.py", i2rt_root / "i2rt/robots/utils.py",
               i2rt_root / "i2rt/robots/config/linear_4310.yml"]
    for sub in ("arm/yam", "gripper/linear_4310"):
        sources += sorted((i2rt_root / "i2rt/robot_models" / sub).rglob("*.xml"))
        sources += sorted((i2rt_root / "i2rt/robot_models" / sub).rglob("*.stl"))
    provenance = {"i2rt_revision": subprocess.check_output(["git", "-C", str(i2rt_root), "rev-parse", "HEAD"], text=True).strip(),
                  "files_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  "combined_model_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
                  "solver": "i2rt.robots.kinematics.Kinematics.ik / mink / quadprog",
                  "site": "grasp_site", "finger_slider_ranges_m": native_limits.tolist()}
    episodes = []
    for ex in sorted((root / "extras").glob("episode_*")):
        rows = [json.loads(l) for l in (ex / "samples.jsonl").read_text().splitlines()]
        images = [r for r in rows if r["kind"] == "image"]
        poses = [r for r in rows if r["kind"] == "pose"]
        ti, tp, _, _ = synchronize_streams(images, poses)
        ti += float(c["camera_time_offset_s"])
        positions = np.asarray([p["position"] for p in poses])
        quats = np.asarray([p["quaternion_xyzw"] for p in poses])
        conf = np.asarray([p["tracker_confidence"] for p in poses])
        raw_tcp = np.asarray([pose_matrix(p, q) @ ext for p, q in zip(positions, quats)])
        dt = np.diff(tp)
        speeds = np.linalg.norm(np.diff(raw_tcp[:, :3, 3], axis=0), axis=1) / dt
        rotations = Rotation.from_matrix(raw_tcp[:, :3, :3])
        angular = (rotations[:-1].inv() * rotations[1:]).magnitude() / dt
        bad = (speeds > c["max_speed_m_s"]) | (angular > c["max_angular_speed_rad_s"]) | (dt > c["max_pose_gap_s"])
        frames, initial, qseed, previous_q, previous_t, previous_width = [], None, seed.copy(), None, None, None
        for i, row in enumerate(images):
            f = {"source_frame": i, "time_s": float(ti[i]), "device_s": row["device_s"], "arrival_s": row["arrival_s"],
                 "pose_valid": False, "width_valid": False, "ik_valid": False, "motion_valid": False,
                 "umi_width_m": 0.0, "yam_open": 0.0, "joint_state": [0.0] * 7,
                 "umi_tcp_world": encode_pose(np.eye(4), 0).tolist(), "yam_ee": encode_pose(np.eye(4), 0).tolist(),
                 "yam_fk": encode_pose(np.eye(4), 0).tolist(), "ik_error": [0.0, 0.0], "reasons": []}
            try:
                measurement = row.get("marker_measurement")
                if measurement and measurement["units"] == "px":
                    width = detector.opening_from_separation(measurement["separation"], c["resolution"])
                elif row["width_measurement"]["valid"]:
                    width = row["width_measurement"]["opening_m"]
                else:
                    raise ValueError(row.get("width_error", "Missing width measurement"))
                f.update(umi_width_m=width, yam_open=normalized_open(width, profile["umi_width_range_m"]), width_valid=True)
            except ValueError as e:
                f["reasons"].append(str(e))
            try:
                tcp = interpolate_pose(tp, positions, quats, conf, ti[i], c["max_pose_gap_s"], c["min_tracker_confidence"]) @ ext
                left = max(0, np.searchsorted(tp, ti[max(0, i-1)], side="right") - 1)
                right = min(len(bad), np.searchsorted(tp, ti[i], side="right"))
                if bad[left:right].any():
                    raise ValueError("Raw tracker discontinuity between observations")
                f["pose_valid"] = True
                if initial is None:
                    initial = np.linalg.inv(tcp)
                target = transform(anchor @ np.linalg.inv(x) @ initial @ tcp @ x)
                f["umi_tcp_world"] = encode_pose(tcp, f["umi_width_m"]).tolist()
                f["yam_ee"] = encode_pose(target, f["yam_open"]).tolist()
                finger_q = native_limits[:, 1] - f["yam_open"] * np.ptp(native_limits, axis=1)
                ok, solution = kin.ik(target, "grasp_site", init_q=np.r_[qseed, finger_q], limits=limits,
                                      max_iters=profile.get("ik_max_iterations", 80), pos_threshold=1e-4, ori_threshold=1e-3)
                solution = np.asarray(solution).copy()
                solution[6:8] = finger_q
                actual = kin.fk(solution).copy()
                err = [float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
                       float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude())]
                if not np.isfinite(solution).all() or not np.isfinite(err).all():
                    raise ValueError("Nonfinite IK solution")
                f["ik_error"] = err
                f["yam_fk"] = encode_pose(actual, f["yam_open"]).tolist()
                f["ik_valid"] = bool(ok and err[0] <= 1e-4 and err[1] <= 1e-3 and
                    np.all(solution[:6] >= model.jnt_range[:6, 0] - 1e-6) and np.all(solution[:6] <= model.jnt_range[:6, 1] + 1e-6))
                if f["ik_valid"]:
                    f["joint_state"] = [*solution[:6].tolist(), f["yam_open"]]
                    f["motion_valid"] = previous_q is not None and previous_t is not None and 0 < ti[i] - previous_t <= 1.5 / c["fps"] and bool(np.max(abs(solution[:6] - previous_q)) / (ti[i] - previous_t) <= profile["max_joint_speed_rad_s"])
                    qseed = solution[:6].copy()
                    previous_q, previous_t = qseed.copy(), ti[i]
                else:
                    previous_q = previous_t = None
                    f["reasons"].append("IK failed FK verification; joint state is an invalid placeholder")
            except ValueError as e:
                previous_q = previous_t = None
                f["reasons"].append(str(e))
            if f["width_valid"] and previous_width is not None and i:
                if abs(f["umi_width_m"] - previous_width) / (ti[i] - ti[i-1]) > c["max_width_speed_m_s"]:
                    f["motion_valid"] = False
                    f["reasons"].append("Aperture speed exceeds capture bound")
            previous_width = f["umi_width_m"] if f["width_valid"] else None
            frames.append(f)
        summary = {key: sum(f[key] for f in frames) for key in ("pose_valid", "width_valid", "ik_valid", "motion_valid")}
        episodes.append({"episode": ex.name, "frames": frames, "quality_counts": summary})
        print(ex.name, len(frames), summary, flush=True)
    doc = {"contract": CONTRACT, "source": str(root), "metadata": meta, "robot_profile": profile, "model": provenance,
           "episodes": episodes, "label_source": "offline i2rt retargeting; not measured robot feedback",
           "registration_verified": False, "hardware_validated": False, "collision_checked": False,
           "gripper_semantics": "UMI metres retained; YAM native_open=(UMI width-closed)/(open-closed), 0 closed and 1 open. Robot physical aperture is not inferred from UMI metres."}
    with output.open("x") as file:
        json.dump(doc, file, indent=2, allow_nan=False)
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw")
    p.add_argument("--profile")
    p.add_argument("--output")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        kin, model, limits, _ = model_setup()
        target = kin.fk(np.zeros(model.nq)).copy()
        ok, _ = kin.ik(target, "grasp_site", init_q=np.zeros(model.nq), limits=limits)
        if not ok:
            raise ValueError("i2rt home FK/IK check failed")
        print("i2rt YAM + linear_4310 FK/IK ready; no hardware connection")
    else:
        annotate(a.raw, json.loads(Path(a.profile).read_text()), a.output)


if __name__ == "__main__":
    main()
