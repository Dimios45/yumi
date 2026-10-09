"""Smooth YAM motion for policy deployment: converged full-pose IK, lift arcs, rate-limited joint streaming.

Why not EEFollower + DecoupledIKSolver for a policy: that pair takes one clipped damped step per
tick toward the latest target. Fine under teleop, where an operator closes the loop, but against
a 30 Hz stream of poses it never converges, drifts between arm configurations and swings the
shoulder, elbow and wrist far more than the tool moves (a logged 12 cm hand path swept j2-j4
by ~130 deg each).

Here every target is solved to convergence, seeded from the previous solution and pulled toward
the posture the arm started in, so consecutive solutions stay on one branch. The joint path
between solutions is streamed at the control rate behind velocity and acceleration limits. A
target far from the current pose is reached through a lift arc (up, across, down) instead of a
straight line through whatever lies between.

Runs in karma's environment (mujoco, vr_teleop_kit).
"""
import threading
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _T(pos, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pos
    return T


class PoseIK:
    """Damped least-squares IK on the tool0 site, solved to convergence, with a posture prior."""

    def __init__(self, q_pref, rot_weight=.3, posture=1e-6, damping=1e-4, margin=.03, iters=80):
        from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver

        self.solver = DecoupledIKSolver()  # used only for its model, tool0 site and FK
        self.model, self.data, self.site = self.solver.model, self.solver.data, self.solver.site_id
        lim = self.solver.joint_limits
        self.lower, self.upper = lim[:, 0]+margin, lim[:, 1]-margin
        self.q_pref = np.clip(np.asarray(q_pref, float)[:6], self.lower, self.upper)
        self.w = np.array([1., 1., 1., rot_weight, rot_weight, rot_weight])
        self.w_position_first = np.array([1., 1., 1., .02, .02, .02])
        self.max_relaxed_deg = 35.  # never trade more orientation than this for position
        # A 6-DoF pose task leaves no null space, so a real posture pull would bias the pose (karma's
        # EEFollower notes the same for its mu). Branch continuity comes from seeding each solve with
        # the previous solution; the prior is only a tie-breaker near singularities/unreachable targets.
        self.mu = posture*np.array([1., 1., 1., .3, .1, .1])
        self.damping, self.iters = damping, iters
        self.jacp, self.jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))

    def fk(self, q):
        pos, quat = self.solver.fk(np.asarray(q, float)[:6])
        return _T(pos, Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix())

    def _error(self, T_target):
        pos = self.data.site_xpos[self.site]
        R = self.data.site_xmat[self.site].reshape(3, 3)
        return np.r_[T_target[:3, 3]-pos, Rotation.from_matrix(T_target[:3, :3] @ R.T).as_rotvec()]

    def solve(self, T_target, q_seed):
        """Returns (q, pos_err_mm, rot_err_deg).

        Exact pose first; if the position cannot be met that way (the orientation is out of reach
        there), solve again with position first and the orientation as close as the arm allows.
        """
        q, mm, deg = self._solve(T_target, q_seed, self.w)
        if mm > 3.:
            q2, mm2, deg2 = self._solve(T_target, q, self.w_position_first)
            if mm2 < mm and deg2 < self.max_relaxed_deg:
                return q2, mm2, deg2
        return q, mm, deg

    def _solve(self, T_target, q_seed, w):
        q = np.clip(np.asarray(q_seed, float)[:6], self.lower, self.upper)
        history = []
        for i in range(self.iters):
            self.solver._fk(q)
            e = self._error(T_target)
            norm = float(np.linalg.norm(e*w))
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            history.append(norm)
            if i >= 8 and norm > .98*history[-6]:
                break  # stalled: the target is out of reach (or at a limit); more iterations won't help
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, :6], self.jacr[:, :6]])*w[:, None]
            A = J.T @ J+np.diag(self.damping+self.mu)
            dq = np.linalg.solve(A, J.T @ (e*w)-self.mu*(q-self.q_pref))
            dq *= min(1., .2/max(np.abs(dq).max(), 1e-9))  # trust region: <= 0.2 rad per iteration
            q = np.clip(q+dq, self.lower, self.upper)
        self.solver._fk(q)
        e = self._error(T_target)
        return q, float(np.linalg.norm(e[:3])*1000), float(np.degrees(np.linalg.norm(e[3:])))


def lift_arc(T_from_, T_to, lift, n):
    """Waypoints from T_from_ to T_to: straight in plan view, raised by `lift` metres mid-way."""
    s = (1-np.cos(np.linspace(0, np.pi, n+1)[1:]))/2  # smooth (cosine) progress, excludes the start
    slerp = Slerp([0, 1], Rotation.from_matrix([T_from_[:3, :3], T_to[:3, :3]]))
    out = []
    for si in s:
        p = (1-si)*T_from_[:3, 3]+si*T_to[:3, 3]
        p[2] += lift*np.sin(np.pi*si)
        out.append(_T(p, slerp(si).as_matrix()))
    return out


class Planner:
    """Targets -> joint knots. Inserts a lift arc when the first target is far from the current pose."""

    def __init__(self, ik, transit_m=.06, lift_m=.06, transit_speed=.12, floor_z=None, max_miss_mm=25.):
        self.ik = ik
        self.transit_m, self.lift_m, self.transit_speed = transit_m, lift_m, transit_speed
        self.floor_z, self.max_miss_mm = floor_z, max_miss_mm

    def plan(self, q_now, targets, due, now):
        """targets (N,4,4) due at absolute times `due`. Returns dict of knot times, joints and diagnostics.

        Targets are raised to `floor_z`. The plan stops before the first target the IK cannot reach
        within `max_miss_mm` (`cut`), so the arm holds instead of stretching after an unreachable pose.
        """
        targets = np.array(targets, float)
        floored = 0
        if self.floor_z is not None:
            floored = int((targets[:, 2, 3] < self.floor_z).sum())
            targets[:, 2, 3] = np.maximum(targets[:, 2, 3], self.floor_z)
        T_now = self.ik.fk(q_now)
        times, Ts, transit = [], [], 0
        gap = np.linalg.norm(targets[0][:3, 3]-T_now[:3, 3])
        turn = Rotation.from_matrix(targets[0][:3, :3] @ T_now[:3, :3].T).magnitude()
        if gap > self.transit_m or turn > np.radians(20):
            gap = max(gap, turn*.15)  # time a large reorientation like a 15 cm/rad move
            duration = max(.8, gap/self.transit_speed)
            arc = lift_arc(T_now, targets[0], self.lift_m, max(8, int(duration*15)))
            start = max(now, due[0]-duration)
            times += list(start+duration*np.arange(1, len(arc)+1)/len(arc))
            Ts += arc
            due = np.asarray(due)+max(0., times[-1]-due[0])  # the chunk resumes after the arc
            times, Ts = times[:-1], Ts[:-1]  # the arc ends exactly on targets[0]
            transit = len(Ts)
        times += list(due)
        Ts += list(targets)
        q, qs, pos, rot, cut = np.asarray(q_now, float), [], [], [], len(Ts)
        for i, T in enumerate(Ts):
            q_new, mm, deg = self.ik.solve(T, q)
            qs.append(q_new)
            pos.append(mm)
            rot.append(deg)
            if mm > self.max_miss_mm and cut == len(Ts):
                cut = i  # keep solving for the preview, but never stream past an unreachable target
            q = q_new
        out = {'times': np.asarray(times), 'q': np.asarray(qs), 'pos_mm': np.asarray(pos),
               'rot_deg': np.asarray(rot), 'transit': transit, 'cut': cut, 'floored': floored,
               'targets': np.asarray(Ts)}
        out['achieved'] = np.array([self.ik.fk(qq) for qq in out['q']])
        out['ok'] = out['pos_mm'] <= 10.
        return out


class JointStreamer:
    """Streams joint commands at `hz` along timed knots, behind velocity and acceleration limits.

    The reference is the knot path linearly interpolated in time (held at the last knot); the
    command follows it with velocity feed-forward plus a proportional correction, then is
    velocity- and acceleration-limited, so the arm never steps or jerks even when a new chunk
    replaces the old one. `send(q, closed)` gets the gripper in EEFollower's 0=open/1=closed.
    """

    def __init__(self, send, hz, q0, closed0, vmax, amax, gripper_speed=2., kp=8.):
        self.send, self.dt = send, 1/hz
        self.vmax, self.amax = np.asarray(vmax, float), np.asarray(amax, float)
        self.gripper_speed, self.kp = gripper_speed, kp
        self.q, self.v = np.asarray(q0, float)[:6].copy(), np.zeros(6)
        self.closed = float(closed0)
        self.lock = threading.Lock()
        now = time.monotonic()
        self.t_knots, self.q_knots, self.g_knots = np.array([now]), self.q[None].copy(), np.array([self.closed])
        self.stop_event = threading.Event()
        self.thread = None

    def plan(self, times, qs, closed):
        """Replace the future: keep the reference continuous from where it is now."""
        now = time.monotonic()
        with self.lock:
            q_ref, g_ref = self._reference(now)
            keep = np.asarray(times) > now
            self.t_knots = np.r_[now, np.asarray(times)[keep]]
            self.q_knots = np.vstack([q_ref[None], np.asarray(qs)[keep]])
            self.g_knots = np.r_[g_ref, np.asarray(closed)[keep]]

    def hold(self):
        with self.lock:
            self.t_knots, self.q_knots, self.g_knots = np.array([time.monotonic()]), self.q[None].copy(), \
                np.array([self.closed])

    def _reference(self, t):
        tk, qk, gk = self.t_knots, self.q_knots, self.g_knots
        if t <= tk[0] or len(tk) == 1:
            return qk[0].copy(), float(gk[0])
        if t >= tk[-1]:
            return qk[-1].copy(), float(gk[-1])
        j = int(np.searchsorted(tk, t))
        a = (t-tk[j-1])/(tk[j]-tk[j-1])
        return (1-a)*qk[j-1]+a*qk[j], float((1-a)*gk[j-1]+a*gk[j])

    def step(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            q_ref, g_ref = self._reference(now)
            q_next, _ = self._reference(now+self.dt)
        v_ff = (q_next-q_ref)/self.dt
        v_des = np.clip(v_ff+self.kp*(q_ref-self.q), -self.vmax, self.vmax)
        # Brake in time: never faster than what can stop within the remaining distance.
        brake = np.sqrt(2*self.amax*np.abs(q_ref-self.q))+np.abs(v_ff)
        v_des = np.clip(v_des, -brake, brake)
        self.v += np.clip(v_des-self.v, -self.amax*self.dt, self.amax*self.dt)
        self.q = self.q+self.v*self.dt
        self.closed += float(np.clip(g_ref-self.closed, -self.gripper_speed*self.dt, self.gripper_speed*self.dt))
        self.send(self.q.copy(), self.closed)
        return self.q.copy()

    def _loop(self):
        next_t = time.monotonic()
        while not self.stop_event.is_set():
            self.step()
            next_t += self.dt
            self.stop_event.wait(max(0., next_t-time.monotonic()))

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)

    @property
    def qpos(self):
        return self.q.copy()

    @property
    def gripper(self):
        return self.closed


def estimate_latency_ms(t, q_commanded, q_measured, max_ms=400):
    """Shift that best aligns measured joints with commanded joints (the arm's execution delay)."""
    t, qc, qm = np.asarray(t), np.asarray(q_commanded), np.asarray(q_measured)
    if len(t) < 20 or np.ptp(qc, axis=0).max() < np.radians(2) or np.ptp(qm, axis=0).max() < np.radians(2):
        return None
    best = None
    for lag in range(0, max_ms+1, 5):
        ok = t+lag/1000 <= t[-1]
        if ok.sum() < 10:
            break
        shifted = np.stack([np.interp(t[ok]+lag/1000, t, qm[:, j]) for j in range(qc.shape[1])], 1)
        err = np.abs(shifted-qc[ok]).mean()
        if best is None or err < best[0]:
            best = (err, lag)
    return best[1] if best else None
