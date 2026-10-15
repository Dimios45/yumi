"""Full-pose differential IK for the YAM, a drop-in for karma's DecoupledIKSolver.

The decoupled solver splits position (joints 1-3) from orientation (joints 4-6).
The YAM wrist axes don't intersect, so wherever the wrist rotates it leaves a
position residual that teleop relies on the operator to close. Replaying UMI
demonstrations has no operator: on IK-rejected frames of session-20261008 that
residual was ~34 mm even after 500 iterations, while a full 6-DoF solve reached
36% of those frames within 10 mm / 5 deg.

This solver keeps karma's model, FK and status attributes (it subclasses the
decoupled solver) and replaces only `solve()`: one mink QP step over all six
joints, with the per-tick joint caps as velocity constraints and the MJCF joint
limits as configuration constraints. One step per call, like the original, so
EEFollower's latest-wins 200 Hz loop is unchanged.
"""
import mink
import mujoco
import numpy as np
from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver

ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]


class MinkIKSolver(DecoupledIKSolver):
    # Defaults from a sweep of filter_training_ik on session-20261008 (10 episodes):
    # orientation weighted 0.1 (1 rad ~ 10 cm) so that at the workspace edge the
    # solver gives up a few degrees rather than centimetres, and a posture pull
    # just strong enough to keep joints out of limit corners. These rejected
    # exactly the 284 frames no IK can reach (decoupled: 457).
    def __init__(self, model_path=None, mu=0.0, max_dq_per_joint=None, orientation_cost=0.1,
                 lm_damping=1.0, posture_cost=0.003, posture=None, limit_margin=0.0,
                 qp_solver='daqp', **kwargs):
        super().__init__(model_path=model_path, mu=mu, max_dq_per_joint=max_dq_per_joint, **kwargs)
        self.configuration = mink.Configuration(self.model)
        self.task = mink.FrameTask('tool0', 'site', position_cost=1.0,
                                   orientation_cost=orientation_cost, lm_damping=lm_damping)
        self.tasks = [self.task]
        # Without a posture preference the six joints drift into joint-limit
        # corners (joint 4 pinned at -pi/2 on session-20261008 episode 9) and
        # stay there. A weak posture task plays the role of the decoupled
        # solver's Tikhonov pull.
        if posture_cost > 0:
            rest = np.zeros(self.model.nq)
            rest[:6] = [0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0] if posture is None else posture
            self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
            self.posture_task.set_target(rest)
            self.tasks.append(self.posture_task)
        # Steps are taken with dt=1, so velocity limits are per-call joint caps.
        self.limits = [mink.ConfigurationLimit(self.model, min_distance_from_limits=limit_margin)]
        if self.max_dq_per_joint is not None:
            self.limits.append(mink.VelocityLimit(
                self.model, dict(zip(ARM_JOINTS, self.max_dq_per_joint))))
        self.qp_solver = qp_solver
        self._J = np.zeros((6, self.model.nv))

    def solve(self, target_pos, target_quat_wxyz, qpos_seed):
        q = np.zeros(self.model.nq)
        q[:6] = np.clip(np.asarray(qpos_seed, float)[:6], self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.configuration.update(q)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(target_quat_wxyz, float)), np.asarray(target_pos, float))
        self.task.set_target(target)
        v = mink.solve_ik(self.configuration, self.tasks, 1.0, self.qp_solver,
                          damping=1e-6, limits=self.limits)
        v[6:] = 0.0
        unclamped = q[:6] + v[:6]
        qpos6 = np.clip(unclamped, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # Status signals the replay viewer and reports already read.
        self._fk(qpos6)
        self.last_pos_err_norm = float(np.linalg.norm(target_pos - self.data.site_xpos[self.site_id]))
        self.last_limit_pressure = float(np.linalg.norm(unclamped - qpos6))
        mujoco.mj_jacSite(self.model, self.data, self._J[:3], self._J[3:], self.site_id)
        sigma = np.linalg.svd(self._J[:, :6], compute_uv=False)
        self.last_singularity_proximity = float(max(0.0, 1.0 - sigma[-1] / 0.05))
        self.last_wrist_gimbal_proximity = float(max(0.0, 1.0 - abs(np.cos(qpos6[4])) / self.w0_rot))
        return qpos6
