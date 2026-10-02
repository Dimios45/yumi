#!/usr/bin/env python3
"""Run the UMI relative-trajectory ACT policy (vruga/yumi-umi-block-place-act-relative).

Run with karma's environment (LeRobot 0.6.1, pyrealsense2, vr_teleop_kit, openpi_control):

    cd ~/karma && uv run python ../yumi/scripts/infer_act_relative.py offline \\
        --checkpoint ../yumi/data/policies/yumi-umi-block-place-act-relative \\
        --dataset ../yumi/data/lerobot/yumi-umi-block-place

Policy contract (matches export_umi_lerobot.py and the checkpoint's train_act_relative.py):

* observation.images.wrist: D405 RGB 480x640, float CHW in [0, 1].
* observation.state (11): TCP pose `obs_step` frames (3 at 30 fps) before the observation,
  relative to the TCP pose at the observation (pos m, rot6d = first two rotation columns),
  then the previous and current jaw width (m).                                   UMI PD2.2
* action (50 x 10): TCP pose at t_obs + (k+1)/fps relative to the TCP pose at t_obs
  (pos, rot6d), absolute jaw width (m).                                          UMI PD2.1

At deployment the TCP is YAM tool0: the dataset expressed the hand in YAM tool0 axes
(R_TRACKER_TCP_ASSUMED). Every chunk is composed with the robot's measured tool0 pose,
interpolated at the image capture time (PD1.1), and steps whose time has already passed
once inference and execution latency are accounted for are discarded (PD1.2).

Modes:
  offline  predict on dataset frames and compare to the recorded relative chunks.
  live     D405 wrist camera + YAM (`--robot yam`) or a virtual arm (`--robot sim`).
"""
import argparse
import bisect
from collections import deque
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

FPS = 30
OBS_STEP = 3  # frames between the two observation poses (10 Hz history at 30 fps)
YAM_FINGER_MAX = .0475  # joint7/joint8 travel; full opening is twice this (replay_yam.py)
MAX_DQ = [.06]*3+[.24]*3  # karma's YAM deployment velocity caps, rad per IK tick


# --------------------------------------------------------------------------- pose maths

def rot6d(rotation):
    return rotation[..., :, :2].swapaxes(-1, -2).reshape(*rotation.shape[:-2], 6)


def rot6d_to_matrix(d6):
    a, b = d6[..., :3], d6[..., 3:6]
    x = a/np.linalg.norm(a, axis=-1, keepdims=True)
    b = b-(x*b).sum(-1, keepdims=True)*x
    y = b/np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def pose_to_T(p):
    T = np.zeros(p.shape[:-1]+(4, 4))
    T[..., :3, :3] = rot6d_to_matrix(p[..., 3:9])
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1
    return T


def relative_state(T_now, T_prev, width_prev, width_now):
    prev = np.linalg.inv(T_now) @ T_prev
    return np.r_[prev[:3, 3], rot6d(prev[:3, :3]), width_prev, width_now].astype(np.float32)


def relative_action_chunk(current_pose10, chunk10):
    rel = np.linalg.inv(pose_to_T(current_pose10))[..., None, :, :] @ pose_to_T(chunk10)
    return np.concatenate([rel[..., :3, 3], rot6d(rel[..., :3, :3]), chunk10[..., 9:10]], -1)


def rotation_deg(Ra, Rb):
    return np.degrees(Rotation.from_matrix(np.swapaxes(Ra, -1, -2) @ Rb).magnitude())


# --------------------------------------------------------------------------- policy

class Policy:
    def __init__(self, checkpoint, device):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA unavailable; running on CPU', flush=True)
            device = 'cpu'
        self.device = device
        self.policy = ACTPolicy.from_pretrained(checkpoint)
        self.policy.config.device = device
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}})
        self.chunk = self.policy.config.chunk_size

    @torch.inference_mode()
    def __call__(self, rgb, state):
        """rgb: HxWx3 uint8 or 3xHxW float [0,1]; state: (11,). Returns the (chunk, 10) relative chunk."""
        image = torch.as_tensor(rgb)
        if image.dtype == torch.uint8:
            image = image.permute(2, 0, 1).float()/255
        batch = self.pre({'observation.images.wrist': image,
                          'observation.state': torch.as_tensor(state, dtype=torch.float32)})
        chunk = self.post(self.policy.predict_action_chunk(batch))
        return chunk[0].double().cpu().numpy()


# --------------------------------------------------------------------------- offline

def offline(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy = Policy(args.checkpoint, args.device)
    root = Path(args.dataset)
    meta = json.loads((root/'meta/info.json').read_text())
    ds = LeRobotDataset(args.repo_id or 'local/yumi-umi', root=root, video_backend='pyav',
                        delta_timestamps={'action': [i/FPS for i in range(policy.chunk)]},
                        episodes=args.episodes)
    indices = range(0, len(ds), args.stride)
    print(f'{len(indices)} frames from {meta["total_episodes"]} episodes '
          f'({"all" if args.episodes is None else args.episodes})', flush=True)
    rows = []
    for n, i in enumerate(indices):
        item = ds[i]
        started = time.perf_counter()
        pred = policy(item['observation.images.wrist'], item['observation.state'].numpy())
        dt = time.perf_counter()-started
        truth = relative_action_chunk(item['observation.ee_pose'].double().numpy(),
                                      item['action'].double().numpy())
        valid = ~item['action_is_pad'].numpy()
        pos = np.linalg.norm(pred[:, :3]-truth[:, :3], axis=-1)[valid]*1000
        rot = rotation_deg(rot6d_to_matrix(pred[:, 3:9]), rot6d_to_matrix(truth[:, 3:9]))[valid]
        width = np.abs(pred[:, 9]-truth[:, 9])[valid]*1000
        rows.append({'index': i, 'episode': int(item['episode_index']), 'frame': int(item['frame_index']),
                     'pos_mm_mean': float(pos.mean()), 'pos_mm_end': float(pos[-1]),
                     'rot_deg_mean': float(rot.mean()), 'width_mm_mean': float(width.mean()),
                     'infer_s': dt})
        if n % 20 == 0:
            r = rows[-1]
            print(f'  [{n}/{len(indices)}] ep {r["episode"]} f {r["frame"]}: pos {r["pos_mm_mean"]:.1f} mm '
                  f'(end {r["pos_mm_end"]:.1f}), rot {r["rot_deg_mean"]:.2f} deg, '
                  f'width {r["width_mm_mean"]:.1f} mm, {dt*1000:.0f} ms', flush=True)
    summary = {k: float(np.mean([r[k] for r in rows]))
               for k in ('pos_mm_mean', 'pos_mm_end', 'rot_deg_mean', 'width_mm_mean', 'infer_s')}
    print('mean over frames: '+json.dumps(summary), flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({'checkpoint': str(args.checkpoint), 'dataset': str(root),
                                                 'summary': summary, 'frames': rows}, indent=1))
        print(f'wrote {args.output}')


# --------------------------------------------------------------------------- live

class PoseBuffer:
    """Timestamped tool0 poses and jaw widths; interpolated at camera capture times (PD1.1)."""

    def __init__(self, seconds=2.):
        self.items = deque(maxlen=int(seconds*500))
        self.lock = threading.Lock()

    def add(self, t, T, width):
        with self.lock:
            if not self.items or t > self.items[-1][0]:
                self.items.append((t, T, width))

    def at(self, t, max_extrapolate=.05):
        with self.lock:
            items = list(self.items)
        if len(items) < 2:
            raise RuntimeError('no robot state yet')
        times = [x[0] for x in items]
        if t < times[0] or t > times[-1]+max_extrapolate:
            raise RuntimeError(f'pose buffer does not cover t={t:.3f} ({times[0]:.3f}..{times[-1]:.3f})')
        j = min(max(bisect.bisect_right(times, t), 1), len(items)-1)
        (ta, Ta, wa), (tb, Tb, wb) = items[j-1], items[j]
        a = float(np.clip((t-ta)/(tb-ta), 0, 1))
        T = np.eye(4)
        T[:3, 3] = (1-a)*Ta[:3, 3]+a*Tb[:3, 3]
        T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([Ta[:3, :3], Tb[:3, :3]]))(a).as_matrix()
        return T, (1-a)*wa+a*wb


class D405:
    """Latest D405 colour frame with its estimated capture time on time.monotonic()."""

    def __init__(self, serial, latency_s):
        import pyrealsense2 as rs

        self.latency_s = latency_s
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, FPS)
        self.pipe.start(cfg)
        for _ in range(30):  # auto-exposure settle
            self.pipe.wait_for_frames()

    def read(self):
        frames = self.pipe.wait_for_frames()
        arrived = time.monotonic()
        return arrived-self.latency_s, np.asanyarray(frames.get_color_frame().get_data()).copy()

    def close(self):
        self.pipe.stop()


def T_from(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    return T


def live(args):
    """Same lifecycle as karma's `inference`: cameras before motors, one try around everything
    after power_up, ctrl-c or any error disarms, then parks at home_pos and de-energizes."""
    from contextlib import ExitStack

    from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver
    from vr_teleop_kit.ik.ee_follower import EEFollower

    for name in ('speed', 'max_step_rad', 'max_effector_step', 'ik_hz'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise SystemExit(f'--{name.replace("_", "-")} must be finite and positive')
    policy = Policy(args.checkpoint, args.device)
    open_width = 2*YAM_FINGER_MAX
    fk_solver = DecoupledIKSolver()  # own MuJoCo data: the follower thread owns the other one
    fk_lock = threading.Lock()

    def fk(q):
        with fk_lock:
            return T_from(*fk_solver.fk(np.asarray(q, float)[:6]))

    buffer = PoseBuffer()
    stop = threading.Event()  # set on ctrl-c, error or end: nothing is commanded after it
    fault = []  # errors raised on background threads, re-raised by the main loop
    session = live_arms = arm = follower = viz = None
    log, failures, failures_out = [], 0, []
    # Camera before the motors, as karma does: a camera held by another process
    # should fail before an arm is energized.
    camera = D405(args.serial, args.camera_latency)
    try:
        if args.robot == 'yam':
            from openpi_control.cli import power_down, power_up, settle_arm_states
            from openpi_control.safety import MAX_STATE_AGE_S, StateGuard, joint_limits, report_gripper_start
            from openpi_control.types import PositionCommand
            from openpi_control.rigs import resolve_rig

            rig = resolve_rig('yam_bimanual').subset([args.arm]).with_interfaces({args.arm: args.interface})
            session, live_arms = power_up(rig)
            arm = live_arms[0].arm
            lower, upper = (v[:6] for v in joint_limits(rig)[args.arm])
            opening = settle_arm_states(live_arms)
            report_gripper_start(opening)
            state = opening[args.arm]
            q0 = np.asarray(state.joints.position_rad, float)[:6]
            open0 = state.effector.position if state.effector is not None else 1.
            guard = StateGuard([args.arm])
            # karma's BoundedChunkExecutor budgets, per 30 Hz tick, spread over the IK ticks.
            step_q, step_e = args.max_step_rad*FPS/args.ik_hz, args.max_effector_step*FPS/args.ik_hz
            last = {'q': q0.copy(), 'e': float(open0)}

            def send(q, closed):  # EEFollower gripper 0=open/1=closed -> karma native 1=open
                if stop.is_set():
                    return
                s = arm.latest_state
                if s is None or not s.is_fresh(MAX_STATE_AGE_S):
                    return  # pause: the node holds its last target; StateGuard fails after 1 s
                q = np.clip(last['q']+np.clip(np.asarray(q, float)-last['q'], -step_q, step_q), lower, upper)
                e = float(np.clip(last['e']+np.clip(1-closed-last['e'], -step_e, step_e), 0, 1))
                last['q'], last['e'] = q, e
                arm.command(PositionCommand(q, effector=e))
        else:
            q0 = np.array(args.start_joints, float)
            open0 = 1.

            def send(q, closed):
                pass

        follower = EEFollower(DecoupledIKSolver(mu=args.mu, max_dq_per_joint=MAX_DQ), send_joints=send,
                              freq=args.ik_hz, q_init=q0, gripper_init=1-open0,
                              gripper_max_speed=args.gripper_speed)

        def state_loop():
            try:
                while not stop.is_set():
                    if arm is not None:
                        s = arm.latest_state
                        if guard.check({args.arm: s}) is not None:
                            width = (s.effector.position if s.effector is not None else 1.)*open_width
                            buffer.add(s.monotonic_timestamp, fk(s.joints.position_rad), width)
                    else:  # virtual arm: the follower's integrated joints are the measurement
                        buffer.add(time.monotonic(), fk(follower.qpos), (1-follower.gripper)*open_width)
                    stop.wait(.004)
            except Exception as err:  # noqa: BLE001 - handed to the main loop, which parks
                fault.append(err)
                stop.set()

        def check():
            if fault:
                raise RuntimeError(f'state monitor: {type(fault[0]).__name__}: {fault[0]}')
            if stop.is_set():
                raise KeyboardInterrupt

        threading.Thread(target=state_loop, daemon=True).start()
        T_start = fk(q0)
        follower.set_target(T_start[:3, 3], Rotation.from_matrix(T_start[:3, :3]).as_quat(), 1-open0)
        follower.start()
        stop.wait(.3)

        if args.viser:
            import viser
            viz = viser.ViserServer(port=args.port)
            print(f'viser    http://localhost:{args.port}', flush=True)

        dt = 1/(FPS*args.speed)
        park = 'park at home_pos and ' if session is not None and args.park else ''
        print(f'running {args.robot} at speed {args.speed}; ctrl-c to {park}stop', flush=True)
        t0 = time.monotonic()
        while args.max_seconds <= 0 or time.monotonic()-t0 < args.max_seconds:
            check()
            t_obs, rgb = camera.read()
            T_now, w_now = buffer.at(t_obs)
            T_prev, w_prev = buffer.at(t_obs-OBS_STEP/FPS)
            started = time.monotonic()
            rel = policy(rgb, relative_state(T_now, T_prev, w_prev, w_now))
            infer_s = time.monotonic()-started
            if not np.isfinite(rel).all():
                raise RuntimeError('non-finite action chunk')
            reach = np.linalg.norm(rel[:, :3], axis=-1).max()
            if reach > args.max_chunk_reach:
                print(f'chunk rejected: reaches {reach*1000:.0f} mm > {args.max_chunk_reach*1000:.0f} mm; '
                      'holding', flush=True)
                continue
            targets = T_now @ pose_to_T(rel)  # PD2.1: compose with the TCP at observation time
            due = t_obs+(np.arange(len(rel))+1)*dt
            # PD1.2: drop steps already outdated once the arm can act on them.
            start = int(np.searchsorted(due, time.monotonic()+args.exec_latency))
            run = range(start, min(start+args.steps_per_inference, len(rel)))
            if viz is not None:
                viz.scene.add_point_cloud('/chunk', points=targets[:, :3, 3].astype(np.float32),
                                          colors=(255, 120, 0), point_size=.006)
                viz.scene.add_frame('/tool0', position=T_now[:3, 3],
                                    wxyz=Rotation.from_matrix(T_now[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                    axes_length=.05, axes_radius=.004)
            print(f'obs w {w_now*1000:4.1f} mm | infer {infer_s*1000:4.0f} ms | skip {start:2d} | '
                  f'run {len(run):2d} | chunk end {np.linalg.norm(rel[-1, :3])*1000:4.0f} mm, '
                  f'width {rel[run[0], 9]*1000 if len(run) else float("nan"):4.1f} mm', flush=True)
            log.append({'t_obs': t_obs, 'infer_s': infer_s, 'skipped': start, 'rel': rel.tolist()})
            for k in run:
                stop.wait(max(0., due[k]-args.exec_latency-time.monotonic()))
                check()
                T = targets[k]
                if T[2, 3] < args.min_z:
                    T = T.copy()
                    T[2, 3] = args.min_z
                closed = float(np.clip(1-rel[k, 9]/open_width, 0, 1))
                follower.set_target(T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat(), closed)
    except KeyboardInterrupt:
        print()
    except Exception as err:  # noqa: BLE001 - reported, then the arm is parked below
        failures += 1
        print(f'inference stopped: {type(err).__name__}: {err}', file=sys.stderr, flush=True)
    finally:
        stop.set()  # disarm first: nothing may race the park

        def quietly(fn):  # a repeated ctrl-c or a failing close must not skip the power-down
            def run():
                try:
                    fn()
                except (Exception, KeyboardInterrupt) as err:  # noqa: BLE001
                    print(f'cleanup: {fn.__qualname__}: {type(err).__name__} {err}', file=sys.stderr)
            return run

        def write_log():
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            Path(args.log).write_text(json.dumps(log))
            print(f'wrote {args.log}')

        with ExitStack() as cleanup:  # LIFO: the power-down registered first runs last, always
            if session is not None:  # parks at home_pos; a ctrl-c during the park de-energizes in place
                cleanup.callback(lambda: failures_out.append(power_down(session, live_arms, park=args.park)))
            cleanup.callback(quietly(camera.close))
            if viz is not None:
                cleanup.callback(quietly(viz.stop))
            if follower is not None:
                cleanup.callback(quietly(follower.stop))
            if args.log:
                cleanup.callback(quietly(write_log))
    return 1 if failures or any(failures_out) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    default_ckpt = Path(__file__).resolve().parents[1]/'data/policies/yumi-umi-block-place-act-relative'
    for p in (off := sub.add_parser('offline')), (on := sub.add_parser('live')):
        p.add_argument('--checkpoint', type=Path, default=default_ckpt,
                       help='local dir or HF repo id (vruga/yumi-umi-block-place-act-relative)')
        p.add_argument('--device', default='cuda')
    off.add_argument('--dataset', type=Path, required=True)
    off.add_argument('--repo-id', default='vruga/yumi-umi-block-place')
    off.add_argument('--episodes', type=int, nargs='*', default=None)
    off.add_argument('--stride', type=int, default=15, help='evaluate every Nth frame')
    off.add_argument('--output', type=Path)

    on.add_argument('--robot', choices=['sim', 'yam'], default='sim')
    on.add_argument('--arm', default='right', help='which arm of the yam_bimanual rig')
    on.add_argument('--interface', default='can1')
    on.add_argument('--serial', default='352122273221', help='wrist D405 serial')
    on.add_argument('--start-joints', type=float, nargs=6, default=[0., .7, .7, 0., 0., 0.],
                    help='sim only: initial joints (rad)')
    on.add_argument('--speed', type=float, default=.5, help='1 = training rate; <1 stretches each chunk')
    on.add_argument('--steps-per-inference', type=int, default=15,
                    help='chunk steps executed before observing again (receding horizon)')
    on.add_argument('--camera-latency', type=float, default=.032,
                    help='capture-to-arrival delay of the D405 frame (s)')
    on.add_argument('--exec-latency', type=float, default=.05, help='command-to-motion delay of the arm (s)')
    on.add_argument('--max-step-rad', type=float, default=.10,
                    help="karma's per-30 Hz-tick joint clamp against the previous command")
    on.add_argument('--max-effector-step', type=float, default=.30,
                    help="karma's per-30 Hz-tick gripper clamp")
    on.add_argument('--max-chunk-reach', type=float, default=.25,
                    help='reject chunks moving the TCP further than this (m)')
    on.add_argument('--min-z', type=float, default=-1., help='floor for tool0 z in the arm base frame (m)')
    on.add_argument('--mu', type=float, default=.005)
    on.add_argument('--ik-hz', type=float, default=200.)
    on.add_argument('--gripper-speed', type=float, default=2.)
    on.add_argument('--max-seconds', type=float, default=0., help='stop after this long (0 = until ctrl-c)')
    on.add_argument('--no-park', dest='park', action='store_false', help='yam: power down in place')
    on.add_argument('--viser', action='store_true')
    on.add_argument('--port', type=int, default=8090)
    on.add_argument('--log', type=Path)
    args = parser.parse_args()
    raise SystemExit((offline if args.mode == 'offline' else live)(args) or 0)


if __name__ == '__main__':
    main()
