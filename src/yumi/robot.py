"""Offline single-arm YAM feasibility checks using Karma's model assembly.

This module never opens a CAN bus. Full-pose IK is checked by FK; success is
not inferred merely from receiving a joint vector. Collision coverage remains
limited to the supplied model and must include the actual scene for deployment.
"""

import importlib.util
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import transform


def pose_error(target, actual):
    return np.r_[
        target[:3, 3] - actual[:3, 3],
        Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec(),
    ]


class KarmaModel:
    def __init__(self, karma_root, arm_xml):
        import mujoco

        self.mj = mujoco
        source = Path(karma_root) / "src/vr_teleop_kit/ik/model.py"
        spec = importlib.util.spec_from_file_location("yumi_karma_model", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model, self.data = module.build_model_with_tool0_site(Path(arm_xml))
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool0")
        grasp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        if grasp < 0:
            raise ValueError("Robot model has no explicit grasp_site")
        if self.model.site_bodyid[self.site] != self.model.site_bodyid[grasp]:
            raise ValueError("tool0 and grasp_site must share a rigid body")
        self.original_tool_position = self.model.site_pos[self.site].copy()
        # Source the contact reference from the selected model, never a universal hardcoded offset.
        self.model.site_pos[self.site] = self.model.site_pos[grasp]
        self.model.site_quat[self.site] = self.model.site_quat[grasp]
        self.tool_correction_m = (
            self.model.site_pos[self.site] - self.original_tool_position
        )
        self.limits = self.model.jnt_range[:6].copy()
        self.collision_geom_count = int(
            np.count_nonzero(self.model.geom_contype | self.model.geom_conaffinity)
        )

    def fk(self, q):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint angles")
        self.data.qpos[:6] = q
        self.mj.mj_forward(self.model, self.data)
        a = np.eye(4)
        a[:3, 3] = self.data.site_xpos[self.site]
        a[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        return transform(a)

    def jacobian(self, q):
        self.fk(q)
        p = np.zeros((3, self.model.nv))
        r = np.zeros_like(p)
        self.mj.mj_jacSite(self.model, self.data, p, r, self.site)
        return np.vstack((p[:, :6], r[:, :6]))

    def penetrations(self, q):
        self.fk(q)

        # The fixed base is fused into world by MuJoCo. Its mesh overlaps link1
        # at the bearing, so the usual parent-child exclusion no longer applies.
        def bearing(c):
            names = {
                self.mj.mj_id2name(
                    self.model,
                    self.mj.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[g]),
                )
                for g in (c.geom1, c.geom2)
            }
            return names == {"base", "link1"}

        return [
            (int(c.geom1), int(c.geom2), float(c.dist))
            for c in self.data.contact
            if c.dist < -0.001 and not bearing(c)
        ]


def solve_ik(model, target, seed, pos_tol=1e-5, rot_tol=1e-4, iterations=200):
    if (
        not np.isfinite([pos_tol, rot_tol]).all()
        or min(pos_tol, rot_tol) <= 0
        or iterations < 1
    ):
        raise ValueError("Positive finite IK tolerances and iterations required")
    target = transform(target)
    q = np.asarray(seed, float).copy()
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Expected finite IK seed")
    if np.any(q < model.limits[:, 0]) or np.any(q > model.limits[:, 1]):
        raise ValueError("Seed violates joint limits")
    weights = np.array([1, 1, 1, 0.15, 0.15, 0.15])
    for _ in range(iterations):
        err = pose_error(target, model.fk(q))
        if np.linalg.norm(err[:3]) <= pos_tol and np.linalg.norm(err[3:]) <= rot_tol:
            return q, {
                "position_error_m": float(np.linalg.norm(err[:3])),
                "rotation_error_rad": float(np.linalg.norm(err[3:])),
            }
        j = weights[:, None] * model.jacobian(q)
        e = weights * err
        dq = np.linalg.solve(j.T @ j + 1e-5 * np.eye(6), j.T @ e)
        dq *= min(1.0, 0.08 / max(np.max(abs(dq)), 1e-12))
        previous = np.linalg.norm(e)
        changed = False
        for scale in [1, 0.5, 0.25, 0.125]:
            candidate = np.clip(q + scale * dq, model.limits[:, 0], model.limits[:, 1])
            if (
                np.linalg.norm(weights * pose_error(target, model.fk(candidate)))
                < previous
            ):
                q = candidate
                changed = True
                break
        if not changed:
            break
    err = pose_error(target, model.fk(q))
    raise ValueError(
        f"IK not reached: {np.linalg.norm(err[:3]) * 1000:.2f} mm, {np.rad2deg(np.linalg.norm(err[3:])):.2f} deg"
    )


def plan_targets(
    model,
    targets,
    times,
    seed,
    max_joint_speed=1.0,
    max_joint_accel=4.0,
    workspace=None,
):
    if (
        not np.isfinite([max_joint_speed, max_joint_accel]).all()
        or min(max_joint_speed, max_joint_accel) <= 0
    ):
        raise ValueError("Positive finite motion limits required")
    targets = [transform(t) for t in targets]
    times = np.asarray(times, float)
    if (
        len(targets) != len(times)
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Need increasing target times and matching trajectory")
    q = np.asarray(seed, float)
    points = []
    residuals = []
    for target in targets:
        if workspace is not None:
            bounds = np.asarray(workspace, float)
            if (
                bounds.shape != (2, 3)
                or np.any(target[:3, 3] < bounds[0])
                or np.any(target[:3, 3] > bounds[1])
            ):
                raise ValueError("Target outside configured workspace")
        q, error = solve_ik(model, target, q)
        if model.penetrations(q):
            raise ValueError("Model collision at target")
        points.append(q.copy())
        residuals.append(error)
    points = np.array(points)
    dt = np.diff(times)
    velocity = np.diff(points, axis=0) / dt[:, None]
    if np.max(abs(velocity)) > max_joint_speed:
        raise ValueError("Trajectory exceeds joint speed limit")
    if (
        len(velocity) > 1
        and np.max(abs(np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]))
        > max_joint_accel
    ):
        raise ValueError("Trajectory exceeds joint acceleration limit")
    # Check intermediate model configurations, not only the sampled targets.
    for a, b in pairwise(points):
        steps = max(2, int(np.ceil(np.max(abs(b - a)) / 0.02)))
        for u in np.linspace(0, 1, steps + 1):
            if model.penetrations((1 - u) * a + u * b):
                raise ValueError("Model collision between targets")
    return {
        "joint_positions_rad": points.tolist(),
        "times_s": (times - times[0]).tolist(),
        "residuals": residuals,
        "tool_reference_correction_m": model.tool_correction_m.tolist(),
        "model_collision_geoms": model.collision_geom_count,
        "collision_exclusions": ["base/link1 bearing overlap"],
        "collision_scope": "Only enabled collision geometries in selected model; no assurance for unmodeled table, objects or cables.",
        "hardware_validated": False,
    }
