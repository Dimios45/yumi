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
MAX_DQ = [.06]*3+[.24]*3  # karma's YAM deployment velocity caps, rad per IK tick (--ik follower)
VMAX = [1.5]*3+[3.]*3  # --ik smooth joint velocity limits (rad/s); karma's per-tick clamp still applies
AMAX = [6.]*3+[12.]*3  # --ik smooth joint acceleration limits (rad/s^2)


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


# Tracker -> TCP rotation the dataset was exported with (export_umi_lerobot.py), with ZERO offset:
# the policy's "TCP" is the T265 itself, carrying YAM tool0 axes.
R_TRACKER_TCP_ASSUMED = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
# Fingertips relative to the T265 in the tracker frame, from the CAD registration
# (configs/umi.yam-collection.json T_tracker_tcp): 9 mm right, 73 mm down, 108 mm forward.
CAD_MOUNT_OFFSET = (.0091, -.0734, -.1076)
DEFAULT_MOUNT_FILE = Path(__file__).resolve().parents[1]/'configs/yam.mount.json'


class Mount:
    """Where the policy's TCP (the T265, tool0 axes) sits on the robot, relative to YAM tool0.

    Same parameters as replay_yam.py: `offset` is the fingertip position in the tracker frame (m),
    `tweak` an extra rotation about the TCP axes (deg). The robot applies the policy's relative
    motion to tool0 @ M and converts the result back to tool0 targets. offset 0, tweak 0 -> M = I.
    """

    def __init__(self, offset=(0., 0., 0.), tweak=(0., 0., 0.)):
        self.lock = threading.Lock()
        self.changed = threading.Event()
        self.set(offset, tweak)

    def set(self, offset, tweak):
        T_tracker_tool = np.eye(4)
        T_tracker_tool[:3, :3] = R_TRACKER_TCP_ASSUMED @ Rotation.from_euler('xyz', tweak, degrees=True).as_matrix()
        T_tracker_tool[:3, 3] = offset
        T_tracker_policy = np.eye(4)
        T_tracker_policy[:3, :3] = R_TRACKER_TCP_ASSUMED
        M = np.linalg.inv(T_tracker_tool) @ T_tracker_policy
        with self.lock:
            self.offset, self.tweak = [float(v) for v in offset], [float(v) for v in tweak]
            self.M, self.M_inv = M, np.linalg.inv(M)
        self.changed.set()

    def to_policy(self, T_tool0):
        with self.lock:
            return T_tool0 @ self.M

    def to_tool0(self, T_policy):
        with self.lock:
            return T_policy @ self.M_inv

    def describe(self):
        return (f'offset {[round(v*100, 1) for v in self.offset]} cm, tweak {[round(v, 1) for v in self.tweak]}°')

    def save(self, path):
        Path(path).write_text(json.dumps({'tcp_offset_m': self.offset, 'mount_tweak_deg': self.tweak,
                                          'frame': 'replay_yam.py convention: fingertips in the T265 frame',
                                          'saved': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))

    @classmethod
    def from_args(cls, args):
        offset, tweak, source = (0., 0., 0.), (0., 0., 0.), 'none (policy TCP = tool0)'
        path = args.mount or (DEFAULT_MOUNT_FILE if DEFAULT_MOUNT_FILE.exists() else None)
        if path is not None:
            saved = json.loads(Path(path).read_text())
            offset, tweak, source = saved['tcp_offset_m'], saved['mount_tweak_deg'], str(path)
        if args.mount_offset is not None:
            offset, source = args.mount_offset, 'command line'
        if args.mount_tweak is not None:
            tweak, source = args.mount_tweak, 'command line'
        mount = cls(offset, tweak)
        mount.source = source
        return mount


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
        self.image_hw = tuple(self.policy.config.input_features['observation.images.wrist'].shape[-2:])

    @torch.inference_mode()
    def __call__(self, rgb, state):
        """rgb: HxWx3 uint8 or 3xHxW float [0,1]; state: (11,). Returns the (chunk, 10) relative chunk."""
        image = torch.as_tensor(rgb)
        if image.dtype == torch.uint8:
            image = image.permute(2, 0, 1).float()/255
        if tuple(image.shape[-2:]) != self.image_hw:  # e.g. train_act_session.py trains at 192x256
            image = torch.nn.functional.interpolate(image[None], size=self.image_hw, mode='bilinear',
                                                    align_corners=False, antialias=True)[0]
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

    def latest(self):
        with self.lock:
            return self.items[-1][1] if self.items else None

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
        found = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()
                 if 'D405' in d.get_info(rs.camera_info.name)]
        if serial not in found:
            raise SystemExit(f'D405 {serial} is not connected; connected D405s: {", ".join(found) or "none"}')
        cfg = rs.config()
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


POS_OK_MM, ROT_OK_DEG = 10., 5.  # replay_yam.py's reach criteria
# Approach-axis pitch below horizontal at the start of the training demos (session-02, from T265 gravity).
DEMO_START_PITCH_DEG = (36, 50)


def T_from(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    return T


def wxyz(R):
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def pose_error(target, actual):
    """(mm, deg) between two 4x4 poses."""
    return (float(np.linalg.norm(target[:3, 3]-actual[:3, 3])*1000),
            float(rotation_deg(target[:3, :3], actual[:3, :3])))


def approach_pitch_deg(T):
    """How far tool0's approach axis (z) points below horizontal."""
    return float(np.degrees(np.arcsin(np.clip(-T[2, 2], -1, 1))))


class IKPreview:
    """Run the deployment follower open-loop through targets: what the IK will command, and its residual.

    Same EEFollower + DecoupledIKSolver, tick rate, velocity caps and reach clamp as the live path;
    each target gets the IK ticks it would get in real time before the next one replaces it.
    """

    def __init__(self, args):
        from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver

        self.solver = DecoupledIKSolver(mu=args.mu, max_dq_per_joint=MAX_DQ)
        self.ik_hz = args.ik_hz
        self.ticks = max(1, round(args.ik_hz/(FPS*args.speed)))

    def __call__(self, q0, targets):
        from vr_teleop_kit.ik.ee_follower import EEFollower

        follower = EEFollower(self.solver, freq=self.ik_hz, q_init=q0)
        out = {'q': [], 'achieved': [], 'pos_mm': [], 'rot_deg': [], 'limit': [], 'singular': []}
        for T in targets:
            follower.set_target(T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())
            limit = singular = 0.
            for _ in range(self.ticks):
                follower.step()
                limit = max(limit, self.solver.last_limit_pressure)
                singular = max(singular, self.solver.last_singularity_proximity)
            A = T_from(*self.solver.fk(follower.qpos))
            mm, deg = pose_error(T, A)
            for key, value in zip(out, (follower.qpos, A, mm, deg, limit, singular)):
                out[key].append(value)
        out = {k: np.asarray(v) for k, v in out.items()}
        out['ok'] = (out['pos_mm'] <= POS_OK_MM) & (out['rot_deg'] <= ROT_OK_DEG)
        return out


class YamMesh:
    """One YAM drawn from the IK model's own meshes (the same geometry replay_yam.py renders).

    Several instances share one MuJoCo model for kinematics: the measured arm solid, the
    commanded and IK-preview arms as translucent ghosts.
    """

    _model = None

    def __init__(self, server, name, color, opacity=None):
        import mujoco
        from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver

        if YamMesh._model is None:
            solver = DecoupledIKSolver()
            YamMesh._model = (mujoco, solver.model, solver.data)
        self.mujoco, self.model, self.data = YamMesh._model
        model = self.model
        self.handles = {}
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            shade = color if opacity is not None else ((45, 55, 70) if gid in (0, 3, 6) else color)
            self.handles[gid] = server.scene.add_mesh_simple(
                f'/{name}/geom_{gid}', vertices=model.mesh_vert[va:va+vn].astype(np.float32),
                faces=model.mesh_face[fa:fa+fn].astype(np.uint32), color=shade, opacity=opacity,
                cast_shadow=opacity is None, receive_shadow=opacity is None)

    def show(self, q, closed=None):
        model, data = self.model, self.data
        data.qpos[:] = 0
        data.qpos[:6] = np.asarray(q, float)[:6]
        if closed is not None and model.nq >= 8:
            data.qpos[6:8] = (1-float(closed))*YAM_FINGER_MAX
        self.mujoco.mj_kinematics(model, data)
        for gid, handle in self.handles.items():
            handle.position = data.geom_xpos[gid].copy()
            handle.wxyz = wxyz(data.geom_xmat[gid].reshape(3, 3))

    def visible(self, value):
        for handle in self.handles.values():
            handle.visible = value


class View:
    """Viser page that separates the failure sources.

    * policy:   orange path = predicted chunk composed with tool0 at the observation.
    * IK:       blue path   = what the IK actually reaches for each target; red dots = targets
                missed by more than 10 mm / 5 deg; orange arm = IK solution at the last executed step.
    * tracking: purple frame/arm = commanded joints; teal frame/arm = measured joints.
    """

    def __init__(self, port, identity, mount=None, mount_editable=False, mount_file=None):
        import viser

        self.server = viser.ViserServer(port=port, label='yumi · YAM policy')
        scene, gui = self.server.scene, self.server.gui
        scene.set_up_direction('+z')
        gui.configure_theme(control_width='medium', brand_color=(39, 135, 173), show_share_button=False)
        scene.add_grid('/floor', width=2., height=2., cell_size=.1, section_size=.5,
                       plane_color=(239, 243, 247), plane_opacity=1, position=(0, 0, -.004))
        scene.add_frame('/base', axes_length=.12, axes_radius=.003)
        self.measured = scene.add_frame('/measured', axes_length=.07, axes_radius=.003)
        self.commanded = scene.add_frame('/commanded', axes_length=.05, axes_radius=.002, visible=False)
        self.tracker = scene.add_frame('/t265', axes_length=.04, axes_radius=.002)
        scene.add_label('/t265/label', 'T265 (policy TCP)')
        self.arm_measured = YamMesh(self.server, 'yam_measured', (180, 192, 202))
        self.arm_commanded = YamMesh(self.server, 'yam_commanded', (150, 60, 200), opacity=.3)
        self.arm_ik = YamMesh(self.server, 'yam_ik', (255, 140, 0), opacity=.25)
        self.arm_commanded.visible(False)
        self.arm_ik.visible(False)
        shown = gui.add_checkbox('Ghost arms (commanded, IK plan)', False)

        @shown.on_update
        def _(_):
            self.ghosts = shown.value
            self.commanded.visible = shown.value
            if not shown.value:
                self.arm_commanded.visible(False)
                self.arm_ik.visible(False)

        self.ghosts = False
        self.cube_moved = threading.Event()
        self.mount = mount
        if mount is not None:
            with gui.add_folder('Mount: fingertips relative to the T265'):
                hint = ('drag to fix the mount; the replayed trajectory redraws' if mount_editable else
                        'read-only while commanding; set it in --shadow and save')
                gui.add_markdown(f'{hint} · CAD: {CAD_MOUNT_OFFSET[0]*100:.1f}, {CAD_MOUNT_OFFSET[1]*100:.1f}, '
                                 f'{CAD_MOUNT_OFFSET[2]*100:.1f} cm')
                offsets = [gui.add_slider(f'offset {a} (cm)', min=-25., max=25., step=.1,
                                          initial_value=round(v*100, 1), disabled=not mount_editable)
                           for a, v in zip('xyz', mount.offset)]
                tweaks = [gui.add_slider(f'tweak r{a} (deg)', min=-45., max=45., step=.5,
                                         initial_value=v, disabled=not mount_editable)
                          for a, v in zip('xyz', mount.tweak)]
                cad = gui.add_button('Use CAD offset', disabled=not mount_editable)
                zero = gui.add_button('Zero (as exported)', disabled=not mount_editable)
                save = gui.add_button(f'Save to {Path(mount_file).name}', disabled=not mount_editable)

                def apply(_=None):
                    mount.set([o.value/100 for o in offsets], [t.value for t in tweaks])

                for handle in offsets+tweaks:
                    handle.on_update(apply)

                @cad.on_click
                def _(_):
                    for o, v in zip(offsets, CAD_MOUNT_OFFSET):
                        o.value = round(v*100, 1)
                    apply()

                @zero.on_click
                def _(_):
                    for handle in offsets+tweaks:
                        handle.value = 0.
                    apply()

                @save.on_click
                def _(_):
                    mount.save(mount_file)
                    print(f'  mount    saved {mount.describe()} to {mount_file}', flush=True)
        self.trail = []
        gui.add_markdown(identity)
        self.image = gui.add_image(np.zeros((240, 320, 3), np.uint8), label='policy input (wrist D405)',
                                   format='jpeg', jpeg_quality=70)
        self.stats = gui.add_markdown('waiting for the first chunk')
        self.tracking = gui.add_markdown('')
        gui.add_markdown('**orange** predicted chunk · **green** steps executed · **blue** IK-reached path · '
                         '**red dots** IK misses (>10 mm / 5°) · **YAM** measured joints · ghosts (optional): '
                         '**purple** commanded, **orange** IK plan · **teal** tool0 trail · axes x red, y green, z blue')

        @self.server.on_client_connect
        def _(client):
            client.camera.up_direction = (0., 0., 1.)
            client.camera.position = (.85, -.75, .6)
            client.camera.look_at = (.25, 0., .2)

    def table(self, z):
        self.server.scene.add_box('/table', color=(214, 196, 168), dimensions=(1.2, 1.2, .01),
                                  position=(.35, 0., z-.005), opacity=.6)

    def cube_marker(self, position, editable):
        self.cube_moved = threading.Event()
        self.cube = self.server.scene.add_transform_controls(
            '/cube', scale=.12, position=tuple(position), disable_rotations=True,
            active_axes=(True, True, False), visible=editable)
        self.server.scene.add_box('/cube/box', color=(200, 40, 40), dimensions=(.03, .03, .03))
        self.cube_fixed = None if editable else np.asarray(position[:2])

        @self.cube.on_update
        def _(_):
            self.cube_moved.set()

    def cube_xy(self):
        return np.asarray(self.cube.position[:2]) if self.cube_fixed is None else self.cube_fixed

    def live(self, T_meas, q_meas, T_cmd=None, q_cmd=None, closed=None):
        scene = self.server.scene
        self.measured.position, self.measured.wxyz = T_meas[:3, 3], wxyz(T_meas[:3, :3])
        if self.mount is not None:
            T_t265 = self.mount.to_policy(T_meas)
            self.tracker.position, self.tracker.wxyz = T_t265[:3, 3], wxyz(T_t265[:3, :3])
        self.arm_measured.show(q_meas, closed)
        self.trail.append(T_meas[:3, 3].astype(np.float32))
        self.trail = self.trail[-300:]
        if len(self.trail) > 1:
            pts = np.asarray(self.trail)
            scene.add_line_segments('/trail', np.stack([pts[:-1], pts[1:]], 1), colors=(0, 170, 170),
                                    thickness=2, thickness_units='screen')
        if T_cmd is not None:
            self.commanded.position, self.commanded.wxyz = T_cmd[:3, 3], wxyz(T_cmd[:3, :3])
            if self.ghosts:
                self.arm_commanded.visible(True)
                self.arm_commanded.show(q_cmd)
                self.commanded.visible = True
            mm, deg = pose_error(T_cmd, T_meas)
            self.tracking.content = (f'**tracking** commanded→measured {mm:.1f} mm, {deg:.1f}° · '
                                     f'**approach pitch** {approach_pitch_deg(T_meas):.0f}° down')
        else:
            self.tracking.content = f'**approach pitch** {approach_pitch_deg(T_meas):.0f}° down'

    def chunk(self, T_now, targets, run, preview, rgb, text):
        scene = self.server.scene
        path = np.vstack([T_now[None, :3, 3], targets[:, :3, 3]]).astype(np.float32)
        scene.add_line_segments('/chunk/path', np.stack([path[:-1], path[1:]], 1), colors=(255, 120, 0),
                                thickness=4, thickness_units='screen')
        if len(run):
            done = path[run.start:run.stop+1]
            scene.add_line_segments('/chunk/run', np.stack([done[:-1], done[1:]], 1), colors=(40, 200, 60),
                                    thickness=8, thickness_units='screen')
        for k in range(0, len(targets), 10):
            scene.add_frame(f'/chunk/frames/{k}', position=targets[k, :3, 3], wxyz=wxyz(targets[k, :3, :3]),
                            axes_length=.025, axes_radius=.0015)
        if preview is not None:
            colors = np.where(preview['ok'][:, None], [[255, 120, 0]], [[230, 0, 0]]).astype(np.uint8)
            scene.add_point_cloud('/chunk/points', points=path[1:], colors=colors, point_size=.01)
            reached = np.vstack([T_now[None, :3, 3], preview['achieved'][:, :3, 3]]).astype(np.float32)
            scene.add_line_segments('/chunk/ik', np.stack([reached[:-1], reached[1:]], 1), colors=(40, 90, 255),
                                    thickness=3, thickness_units='screen')
            last = (run.stop-1) if len(run) else len(targets)-1
            if self.ghosts:
                self.arm_ik.visible(True)
                self.arm_ik.show(preview['q'][min(last, len(preview['q'])-1)])
        if rgb is not None:
            self.image.image = rgb[::2, ::2]
        self.stats.content = text

    def stop(self):
        self.server.stop()


def load_demo(dataset, episode):
    """Recorded TCP trajectory of one episode, relative to its first frame, and its jaw widths."""
    import pandas as pd

    df = pd.read_parquet(Path(dataset)/'data/chunk-000/file-000.parquet')
    ep = df[df['episode_index'] == episode]
    if not len(ep):
        raise SystemExit(f'episode {episode} not in {dataset} ({df["episode_index"].nunique()} episodes)')
    ee = np.stack(ep['observation.ee_pose'].values).astype(float)
    T = pose_to_T(ee)
    return np.linalg.inv(T[0]) @ T, ee[:, 9]


def load_session(args):
    """Raw capture of one session episode as fingertip (TCP) poses in a gravity frame (z up).

    Same conventions as visualize_session_ik.py: replay_yam.load_capture at 30 Hz, the CAD
    T_tracker_tcp from --tcp-config with an optional roll about the approach axis, T265 world
    mapped to z-up. Heading is turned so the first approach direction points along robot +X.
    """
    import replay_yam as replay

    root = args.session/'extras'/f'episode_{args.episode:06d}'
    if not root.exists():
        raise SystemExit(f'{root} not found; episodes: {", ".join(p.name for p in sorted(root.parent.glob("episode_*")))}')
    cfg = json.loads(args.tcp_config.read_text())
    ext = np.asarray(cfg['T_tracker_tcp'], float) @ replay.homogeneous(
        Rotation.from_euler('z', args.tool_roll_deg, degrees=True).as_matrix())
    meta, images, times, grid, world_tracker, valid = replay.load_capture(root, 3, .025, FPS)
    if not valid.any():
        raise SystemExit(f'{root.name}: no confidently tracked samples')
    keep = np.arange(np.argmax(valid), len(valid)-np.argmax(valid[::-1]))
    tcp = replay.homogeneous(replay.T265_TO_BASE) @ world_tracker[keep] @ ext
    yaw = -np.arctan2(tcp[0][1, 2], tcp[0][0, 2])
    tcp = replay.homogeneous(Rotation.from_euler('z', yaw).as_matrix()) @ tcp
    image_t = np.array([r['device_s'] for r in images])
    width_ok = np.array([bool(r.get('width_measurement', {}).get('valid')) for r in images])
    widths = (np.interp(grid[keep], image_t[width_ok],
                        [r['width_measurement']['opening_m'] for r, ok in zip(images, width_ok) if ok])
              if width_ok.any() else np.full(len(keep), YAM_FINGER_MAX*2))
    # Pick = lowest fingertip point: the block is picked off the table and placed on top of another,
    # so the place is higher. (The jaw-width signal is too late/noisy in this session to time it.)
    grasp = int(np.argmin(tcp[:, 2, 3]))
    task = meta.get('config', {}).get('task', '')
    return {'tcp': tcp, 'widths': widths, 'grasp': grasp, 'valid': valid[keep], 'task': task,
            'name': root.name, 'untracked': int((~valid[keep]).sum())}


def session_targets(sess, T_start_tool0, args, cube_xy=None):
    """Place the recorded fingertip path in the robot base frame.

    gravity (default): keep the hand's real tilt and up/down; translate so the grasp lands at
      `cube_xy` (or straight below/above the start-relative grasp point) at table height + grasp height.
    relative: the motion in the tool frame from the robot's current pose (what the policy does);
      only reaches the cube if the robot starts where the hand started relative to it.
    """
    tcp = sess['tcp']
    if args.mapping == 'relative':
        return T_start_tool0 @ np.linalg.inv(tcp[0]) @ tcp
    shift = T_start_tool0[:3, 3]-tcp[0][:3, 3]  # start where the arm is ...
    g = tcp[sess['grasp']][:3, 3]+shift
    if cube_xy is not None:
        shift[:2] += np.asarray(cube_xy)-g[:2]  # ... or put the grasp on the cube
    shift[2] = args.table_z+args.grasp_height-tcp[sess['grasp']][2, 3]  # grasp at table height
    out = tcp.copy()
    out[:, :3, 3] += shift
    return out


def preview_text(preview, run):
    sel = slice(run.start, run.stop) if len(run) else slice(None)
    if not len(preview['pos_mm'][sel]):
        sel = slice(None)
    return (f'**IK** max miss {preview["pos_mm"][sel].max():.1f} mm / {preview["rot_deg"][sel].max():.1f}° '
            f'on executed steps · {int((~preview["ok"]).sum())}/{len(preview["ok"])} targets missed · '
            + (f'limit pressure {preview["limit"].max():.2f} · singularity {preview["singular"].max():.2f}'
               if 'limit' in preview else
               f'lift arc {preview["transit"]} knots · floored {preview["floored"]} · '
               + ('all reachable' if preview['cut'] >= len(preview['pos_mm'])
                  else f'stops at {preview["cut"]-preview["transit"]} (unreachable)')))


def live(args):
    """Same lifecycle as karma's `inference`: cameras before motors, one try around everything
    after power_up, ctrl-c or any error disarms, then parks at home_pos and de-energizes.

    --shadow never commands the arm (it holds, or floats with --float): predictions, IK preview and
    measured pose only. --source demo replays a recorded episode's TCP trajectory instead of the policy.
    """
    from contextlib import ExitStack

    from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver
    from vr_teleop_kit.ik.ee_follower import EEFollower
    from yam_motion import JointStreamer, Planner, PoseIK, estimate_latency_ms

    for name in ('speed', 'max_step_rad', 'max_effector_step', 'ik_hz'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise SystemExit(f'--{name.replace("_", "-")} must be finite and positive')
    if args.float and not (args.shadow and args.robot == 'yam'):
        raise SystemExit('--float is only for --robot yam --shadow (a floating arm cannot follow commands)')
    policy = Policy(args.checkpoint, args.device) if args.source == 'policy' else None
    demo = (load_demo(args.dataset, args.episode) if args.source == 'demo' else
            load_session(args) if args.source == 'session' else None)
    open_width = 2*YAM_FINGER_MAX
    fk_solver = DecoupledIKSolver()  # own MuJoCo data: the follower thread owns the other one
    fk_lock = threading.Lock()

    def fk(q):
        with fk_lock:
            return T_from(*fk_solver.fk(np.asarray(q, float)[:6]))

    identity = (f'**arm** {args.arm} on `{args.interface}` ({args.robot}{", shadow" if args.shadow else ""}'
                f'{", float" if args.float else ""}) · **camera** D405 `{args.serial}` · **source** '
                + (f'policy `{Path(str(args.checkpoint)).name}` on {policy.device}' if policy else
                   f'session `{args.session.name}` {demo["name"]} ({args.mapping}, grasp on the table)'
                   if args.source == 'session' else f'demo episode {args.episode}')
                + f' · **speed** {args.speed}')
    mount = Mount.from_args(args)
    identity += (f' · **TCP** CAD T_tracker_tcp from {args.tcp_config.name} (mount file not used)'
                 if args.source == 'session' else f' · **mount** {mount.describe()} ({mount.source})')
    print(identity.replace('**', '').replace('`', ''), flush=True)

    buffer = PoseBuffer()
    stop = threading.Event()  # set on ctrl-c, error or end: nothing is commanded after it
    fault = []  # errors raised on background threads, re-raised by the main loop
    measured_q = {'q': None}
    measured_closed = [None]
    session = live_arms = arm = follower = viz = None
    log, failures, failures_out = [], 0, []
    # Camera before the motors, as karma does: a camera held by another process
    # should fail before an arm is energized.
    camera = D405(args.serial, args.camera_latency) if policy is not None else None
    try:
        if args.robot == 'yam':
            from openpi_control.cli import power_down, power_up, settle_arm_states
            from openpi_control.safety import MAX_STATE_AGE_S, StateGuard, joint_limits, report_gripper_start
            from openpi_control.types import PositionCommand
            from openpi_control.rigs import resolve_rig

            rig = resolve_rig('yam_bimanual').subset([args.arm]).with_interfaces({args.arm: args.interface})
            session, live_arms = power_up(rig, float_mode=args.float)
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
                if stop.is_set() or args.shadow:
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

        if args.ik == 'smooth':
            planner = Planner(PoseIK(q_pref=q0), transit_m=args.transit, lift_m=args.lift,
                              floor_z=args.min_z if args.min_z > -1 else None, max_miss_mm=args.max_ik_miss)
            follower = JointStreamer(send, args.ik_hz, q0, 1-open0, VMAX, AMAX, gripper_speed=args.gripper_speed)
        else:
            planner = None
            follower = EEFollower(DecoupledIKSolver(mu=args.mu, max_dq_per_joint=MAX_DQ), send_joints=send,
                                  freq=args.ik_hz, q_init=q0, gripper_init=1-open0,
                                  gripper_max_speed=args.gripper_speed)

        def state_loop():
            try:
                while not stop.is_set():
                    if arm is not None:
                        s = arm.latest_state
                        if guard.check({args.arm: s}) is not None:
                            q = np.asarray(s.joints.position_rad, float)[:6]
                            measured_q['q'] = q
                            measured_closed[0] = 1-(s.effector.position if s.effector is not None else 1.)
                            width = (s.effector.position if s.effector is not None else 1.)*open_width
                            buffer.add(s.monotonic_timestamp, fk(q), width)
                    else:  # virtual arm: the follower's integrated joints are the measurement
                        measured_q['q'] = follower.qpos
                        measured_closed[0] = follower.gripper
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

        commanding = not args.shadow
        threading.Thread(target=state_loop, daemon=True).start()
        T_start = fk(q0)
        if commanding:
            if planner is None:
                follower.set_target(T_start[:3, 3], Rotation.from_matrix(T_start[:3, :3]).as_quat(), 1-open0)
            follower.start()
        stop.wait(.3)
        pitch = approach_pitch_deg(T_start)
        lo, hi = DEMO_START_PITCH_DEG
        print(f'  start    tool0 approach axis {pitch:.0f}° below horizontal (demos started at {lo}-{hi}°)'
              + ('' if lo-15 <= pitch <= hi+15 else '  <-- the wrist camera will not see what training saw'),
              flush=True)

        follower_preview = IKPreview(args) if planner is None and (args.viser or args.shadow) else None
        if args.viser:
            viz = View(args.port, identity, mount, mount_editable=not commanding,
                       mount_file=args.mount or DEFAULT_MOUNT_FILE)
            print(f'viser    http://<this-machine-ip>:{args.port} (hostname -I)', flush=True)

        def show_live():
            if viz is None or (q := measured_q['q']) is None:
                return
            T_meas = fk(q)
            if commanding:
                viz.live(T_meas, q, fk(follower.qpos), follower.qpos, measured_closed[0])
            else:
                viz.live(T_meas, q, closed=measured_closed[0])

        plan_s = [.03]  # running estimate of planning time (s)

        def plan_chunk(targets, due, run):
            """What the motion layer will do with targets[start:], for preview and execution.

            Returns (plan, shown_targets, shown_run): the plan is aligned with shown_targets, which
            includes a lift arc in front when the smooth planner inserted one.
            """
            q_ref = follower.qpos if commanding else measured_q['q']
            if commanding:  # plan what will run plus a short look-ahead for the preview
                targets, due = targets[:run.stop+15], due[:run.stop+15]
            if planner is not None:
                started = time.monotonic()
                plan = planner.plan(q_ref, targets, np.asarray(due)-args.exec_latency, started)
                plan_s[0] = .7*plan_s[0]+.3*(time.monotonic()-started)
                n = plan['transit']
                return plan, plan['targets'], range(n+run.start, n+run.stop)
            if follower_preview is None:
                return None, targets, run
            return follower_preview(q_ref, targets), targets, run

        def execute(targets, widths, due, run, entry, plan):
            closed = np.clip(1-np.asarray(widths)/open_width, 0, 1)
            if planner is None:
                for k in run:
                    stop.wait(max(0., due[k]-args.exec_latency-time.monotonic()))
                    check()
                    T = targets[k]
                    if T[2, 3] < args.min_z:
                        T = T.copy()
                        T[2, 3] = args.min_z
                    follower.set_target(T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat(), float(closed[k]))
                    if (q := measured_q['q']) is not None:
                        entry['steps'].append({'k': int(k), 't': time.monotonic(), 'target': T.tolist(),
                                               'commanded_q': follower.qpos.tolist(), 'measured_q': q.tolist()})
                    show_live()
                return
            n_transit = plan['transit']
            last = min(plan['cut'], n_transit+run.stop)
            if last <= n_transit+run.start and plan['cut'] < len(plan['times']):
                where = (f'target {plan["cut"]-n_transit}' if plan['cut'] >= n_transit
                         else 'the lift arc')
                print(f'  holding: {where} unreachable ({plan["pos_mm"][plan["cut"]]:.0f} mm IK miss)', flush=True)
            knots = range(0, last)
            gripper = np.r_[np.full(n_transit, closed[run.start] if len(run) else follower.gripper), closed]
            follower.plan(plan['times'][knots.start:knots.stop], plan['q'][knots.start:knots.stop],
                          gripper[knots.start:knots.stop])
            for i in range(max(n_transit, n_transit+run.start), last):
                stop.wait(max(0., plan['times'][i]-time.monotonic()))
                check()
                if (q := measured_q['q']) is not None:
                    entry['steps'].append({'k': int(i-n_transit), 't': time.monotonic(),
                                           'target': plan['targets'][i].tolist(),
                                           'commanded_q': follower.qpos.tolist(), 'measured_q': q.tolist()})
                show_live()
            if last < n_transit+run.stop:  # cut short: hold where the plan ended until the next chunk
                stop.wait(max(0., due[min(run.stop, len(due))-1]-args.exec_latency-time.monotonic()))

        dt = 1/(FPS*args.speed)
        park = 'park at home_pos and ' if session is not None and args.park else ''
        print(f'running ({"shadow: no commands" if args.shadow else "commanding"}, ik {args.ik}, '
              f'exec latency {args.exec_latency*1000:.0f} ms); ctrl-c to {park}stop', flush=True)
        t0 = time.monotonic()

        def running():
            return args.max_seconds <= 0 or time.monotonic()-t0 < args.max_seconds

        if demo is not None:
            T_start_tool0 = buffer.latest()
            if args.source == 'session':
                widths = np.r_[demo['widths'][0], demo['widths']]
                if viz is not None:
                    viz.table(args.table_z)
                    viz.cube_marker(session_targets(demo, T_start_tool0, args)[demo['grasp']][:3, 3],
                                           editable=not commanding)
            else:
                T_ep, widths = demo

            def demo_plan():
                if args.source == 'session':
                    cube_xy = viz.cube_xy() if viz is not None else args.cube
                    targets = session_targets(demo, T_start_tool0, args, cube_xy)
                else:
                    # The demo moves the policy TCP (T265); the robot moves tool0 so that point follows it.
                    T_policy = mount.to_policy(T_start_tool0)
                    targets = mount.to_tool0(T_policy @ T_ep[1:])
                due = time.monotonic()+.5+args.exec_latency+np.arange(len(targets))*dt
                run = range(0, len(targets))
                pv, shown, shown_run = plan_chunk(targets, due, run)
                reach = np.linalg.norm(targets[:, :3, 3]-T_start_tool0[:3, 3], axis=1).max()
                if args.source == 'session':
                    g = targets[demo['grasp']][:3, 3]
                    text = (f'**{demo["name"]}** ({args.mapping}) "{demo["task"].strip()}": {len(targets)} steps · '
                            f'grasp at step {demo["grasp"]} → ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.3f}) m · '
                            f'lowest {targets[:, 2, 3].min():.3f} m · table {args.table_z:.3f} m · '
                            f'starts {np.linalg.norm(targets[0][:3, 3]-T_start_tool0[:3, 3])*1000:.0f} mm from the arm')
                else:
                    text = (f'**demo episode {args.episode}**: {len(targets)} steps · T265 path '
                            f'{np.linalg.norm(T_ep[-1, :3, 3])*1000:.0f} mm net · fingertips reach '
                            f'{reach*1000:.0f} mm from start · mount {mount.describe()}')
                text = (text
                        + (f'\n\n{preview_text(pv, shown_run)}' if pv is not None else ''))
                print(text.replace('**', '').replace('\n\n', ' | '), flush=True)
                if viz is not None:
                    viz.chunk(T_start_tool0, shown, shown_run, pv, None, text)
                return targets, due, run, pv

            mount.changed.clear()
            targets, due, run, pv = demo_plan()
            entry = {'source': args.source, 'start': T_start_tool0.tolist(), 'mount': [mount.offset, mount.tweak],
                     'steps': []}
            log.append(entry)
            if commanding:
                execute(targets, widths[1:], due, run, entry, pv)
            while running():
                check()
                if (mount.changed.is_set() or (viz is not None and viz.cube_moved.is_set())) and not commanding:
                    mount.changed.clear()
                    if viz is not None:
                        viz.cube_moved.clear()
                    demo_plan()
                show_live()
                stop.wait(.05)
            return 0

        if args.start_from_session is not None:
            # Start like the demonstrations: the hand's start pose from a session episode, gravity-mapped
            # with its pick on the table (and on --cube if given), so the wrist camera sees a training view.
            import copy
            sa = copy.copy(args)
            sa.episode, sa.mapping = args.start_from_session, 'gravity'
            sess = load_session(sa)
            T_goal = session_targets(sess, buffer.latest(), sa, args.cube)[0]
            print(f'  start    {sess["name"]} start pose: ({T_goal[0, 3]:.2f}, {T_goal[1, 3]:.2f}, {T_goal[2, 3]:.2f}) m, '
                  f'approach {approach_pitch_deg(T_goal):.0f}° down', flush=True)
            if viz is not None:
                viz.table(args.table_z)
                viz.server.scene.add_frame('/start_goal', position=T_goal[:3, 3], wxyz=wxyz(T_goal[:3, :3]),
                                           axes_length=.06, axes_radius=.004)
            if commanding:
                if planner is None:
                    raise SystemExit('--start-from-session needs --ik smooth')
                now = time.monotonic()
                plan = planner.plan(follower.qpos, T_goal[None], [now+.5], now)
                if plan['cut'] < len(plan['times']):
                    raise RuntimeError(f'start pose unreachable ({plan["pos_mm"][plan["cut"]]:.0f} mm IK miss); '
                                       'move the arm closer or pass --cube')
                closed = float(np.clip(1-sess['widths'][0]/open_width, 0, 1))
                follower.plan(plan['times'], plan['q'], np.full(len(plan['times']), closed))
                while time.monotonic() < plan['times'][-1]+1.:  # arrive, then settle
                    check()
                    show_live()
                    stop.wait(.05)
                print('  start    reached; policy takes over', flush=True)

        while running():
            check()
            t_obs, rgb = camera.read()
            T_now, w_now = buffer.at(t_obs)
            T_prev, w_prev = buffer.at(t_obs-OBS_STEP/FPS)
            # The policy sees and moves its own TCP (the T265 point on the robot, via the mount).
            P_now, P_prev = mount.to_policy(T_now), mount.to_policy(T_prev)
            started = time.monotonic()
            rel = policy(rgb, relative_state(P_now, P_prev, w_prev, w_now))
            infer_s = time.monotonic()-started
            if not np.isfinite(rel).all():
                raise RuntimeError('non-finite action chunk')
            targets = mount.to_tool0(P_now @ pose_to_T(rel))  # PD2.1: compose with the TCP at observation time
            due = t_obs+(np.arange(len(rel))+1)*dt
            # PD1.2: drop steps already outdated once the arm can act on them, counting the
            # planning that still has to happen before the first command goes out.
            start = int(np.searchsorted(due, time.monotonic()+args.exec_latency+plan_s[0]+.01))
            if start >= len(rel):
                continue
            run = range(0, min(args.steps_per_inference, len(rel)-start)) if commanding else range(0)
            pv, shown, shown_run = plan_chunk(targets[start:], due[start:], run)
            reach = np.linalg.norm(rel[:, :3], axis=-1).max()
            d = rel[-1, :3]*1000
            text = (f'**inference** {infer_s*1000:.0f} ms · **obs→now** {(time.monotonic()-t_obs)*1000:.0f} ms · '
                    f'**plan** {plan_s[0]*1000:.0f} ms · **skipped** {start} · **executing** {len(run)}\n\n'
                    f'**chunk end in tool frame** x {d[0]:+.0f} y {d[1]:+.0f} z(approach) {d[2]:+.0f} mm · '
                    f'**in base** Δz {(targets[-1, 2, 3]-T_now[2, 3])*1000:+.0f} mm\n\n'
                    f'**width** now {w_now*1000:.1f} mm → {rel[-1, 9]*1000:.1f} mm'
                    + (f'\n\n{preview_text(pv, shown_run)}' if pv is not None else ''))
            print(text.replace('**', '').replace('\n\n', ' | '), flush=True)
            if viz is not None:
                viz.chunk(T_now, shown, shown_run, pv, rgb, text)
            entry = {'t_obs': t_obs, 'infer_s': infer_s, 'skipped': start, 'T_obs': T_now.tolist(),
                     'rel': rel.tolist(), 'steps': []}
            if pv is not None:
                entry.update(ik_pos_mm=pv['pos_mm'].tolist(), ik_rot_deg=pv['rot_deg'].tolist())
            log.append(entry)
            if reach > args.max_chunk_reach:
                print(f'chunk rejected: reaches {reach*1000:.0f} mm > {args.max_chunk_reach*1000:.0f} mm; '
                      'holding', flush=True)
                continue
            if commanding:
                execute(targets[start:], rel[start:, 9], due[start:], run, entry, pv)
            else:
                show_live()
                stop.wait(.2)
    except KeyboardInterrupt:
        print()
    except Exception as err:  # noqa: BLE001 - reported, then the arm is parked below
        failures += 1
        print(f'inference stopped: {type(err).__name__}: {err}', file=sys.stderr, flush=True)
    finally:
        stop.set()  # disarm first: nothing may race the park
        steps = [st for entry in log for st in entry.get('steps', [])]
        if not args.shadow and steps:
            lag = estimate_latency_ms([st['t'] for st in steps], [st['commanded_q'] for st in steps],
                                      [st['measured_q'] for st in steps])
            if lag is not None:
                print(f'  latency  measured joints trail commanded by ~{lag} ms this run '
                      f'(--exec-latency was {args.exec_latency*1000:.0f} ms)', flush=True)
                log.append({'measured_exec_latency_ms': lag})

        def quietly(fn):  # a repeated ctrl-c or a failing close must not skip the power-down
            def run():
                try:
                    fn()
                except (Exception, KeyboardInterrupt) as err:  # noqa: BLE001
                    print(f'cleanup: {fn.__qualname__}: {type(err).__name__} {err}', file=sys.stderr)
            return run

        def write_log():
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            Path(args.log).write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()},
                                                  'chunks': log}))
            print(f'wrote {args.log}')

        with ExitStack() as cleanup:  # LIFO: the power-down registered first runs last, always
            if session is not None:  # parks at home_pos; a ctrl-c during the park de-energizes in place
                cleanup.callback(lambda: failures_out.append(power_down(session, live_arms, park=args.park)))
            if camera is not None:
                cleanup.callback(quietly(camera.close))
            if viz is not None:
                cleanup.callback(quietly(viz.stop))
            if follower is not None and not args.shadow:
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
    on.add_argument('--shadow', action='store_true',
                    help='never command the arm: show predictions, IK preview and the measured pose only')
    on.add_argument('--float', action='store_true',
                    help='with --robot yam --shadow: gravity float, so the arm can be moved by hand')
    on.add_argument('--source', choices=['policy', 'demo', 'session'], default='policy',
                    help='demo: replay a training-dataset episode relative to the start pose; '
                         'session: replay a raw yumi session episode (--session), gravity-mapped')
    on.add_argument('--session', type=Path,
                    default=Path(__file__).resolve().parents[1]/'data/lerobot/session-20261008-000612')
    on.add_argument('--mapping', choices=['gravity', 'relative'], default='gravity',
                    help='session: gravity keeps the hand tilt and puts the grasp on the table; '
                         'relative replays tool-frame motion from wherever the arm starts')
    on.add_argument('--start-from-session', type=int, metavar='EPISODE',
                    help='policy: first move the arm to where the hand started in this --session episode '
                         '(gravity-mapped, pick on the table / --cube), then hand over to the policy')
    on.add_argument('--cube', type=float, nargs=2, metavar=('X', 'Y'),
                    help='session gravity: cube position in the arm base frame (m); drag it in viser in --shadow')
    on.add_argument('--table-z', type=float, default=0.,
                    help='table surface height in the arm base frame (m); configs/world_camera.json has base z 0')
    on.add_argument('--grasp-height', type=float, default=.015,
                    help='fingertip midpoint above the table at the grasp (m), about half the cube height')
    on.add_argument('--tcp-config', type=Path,
                    default=Path(__file__).resolve().parents[1]/'data/lerobot/yumi-block-stack-20261008-training/meta/processing-config.json',
                    help='session: config with T_tracker_tcp. Default: the training export, so replays use the same '
                         'fingertip TCP and YAM tool axes the policy is trained on (CAD offset, CAD rotation turned '
                         '90° about the approach axis)')
    on.add_argument('--tool-roll-deg', type=float, default=0.,
                    help='session: capture TCP to YAM tool rotation about the approach axis (deg)')
    on.add_argument('--episode', type=int, default=0, help='--source demo: dataset episode index')
    on.add_argument('--mount', type=Path,
                    help=f'mount JSON (tcp_offset_m, mount_tweak_deg); default {DEFAULT_MOUNT_FILE.name} if present')
    on.add_argument('--mount-offset', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                    help=f'fingertips in the T265 frame (m), replay_yam --tcp-offset; CAD {CAD_MOUNT_OFFSET}')
    on.add_argument('--mount-tweak', type=float, nargs=3, metavar=('RX', 'RY', 'RZ'),
                    help='extra mount rotation about the TCP axes (deg), replay_yam --mount-tweak')
    on.add_argument('--dataset', type=Path,
                    default=Path(__file__).resolve().parents[1]/'data/lerobot/yumi-umi-block-place')
    on.add_argument('--arm', default='right', help='which arm of the yam_bimanual rig')
    on.add_argument('--interface', default='can_right', help="CAN interface ('ip -brief link show type can')")
    on.add_argument('--serial', default='352122271723',
                    help='wrist D405 serial (352122271723 right wrist, 409122274921 left wrist on this rig)')
    on.add_argument('--start-joints', type=float, nargs=6, default=[0., .7, .7, 0., 0., 0.],
                    help='sim only: initial joints (rad)')
    on.add_argument('--speed', type=float, default=.5, help='1 = training rate; <1 stretches each chunk')
    on.add_argument('--steps-per-inference', type=int, default=15,
                    help='chunk steps executed before observing again (receding horizon)')
    on.add_argument('--camera-latency', type=float, default=.032,
                    help='capture-to-arrival delay of the D405 frame (s)')
    on.add_argument('--exec-latency', type=float, default=.10,
                    help='command-to-motion delay of the arm (s); 70-100 ms measured on this YAM, '
                         'each run prints its own estimate at the end')
    on.add_argument('--ik', choices=['smooth', 'follower'], default='smooth',
                    help='smooth: converged pose IK + lift arcs + rate-limited joint streaming (yam_motion.py); '
                         "follower: karma's EEFollower/DecoupledIKSolver as before")
    on.add_argument('--lift', type=float, default=.06, help='smooth: height of the lift arc for far targets (m)')
    on.add_argument('--transit', type=float, default=.06,
                    help='smooth: a target further than this from the arm is reached through a lift arc (m)')
    on.add_argument('--max-ik-miss', type=float, default=25.,
                    help='smooth: stop the chunk before a target the IK misses by more than this (mm)')
    on.add_argument('--max-step-rad', type=float, default=.10,
                    help="karma's per-30 Hz-tick joint clamp against the previous command")
    on.add_argument('--max-effector-step', type=float, default=.30,
                    help="karma's per-30 Hz-tick gripper clamp")
    on.add_argument('--max-chunk-reach', type=float, default=.25,
                    help='reject chunks moving the TCP further than this (m)')
    on.add_argument('--min-z', type=float, default=None,
                    help='floor for tool0 z in the arm base frame (m); default table height + 1 cm; -1 disables')
    on.add_argument('--mu', type=float, default=.005)
    on.add_argument('--ik-hz', type=float, default=200.)
    on.add_argument('--gripper-speed', type=float, default=2.)
    on.add_argument('--max-seconds', type=float, default=0., help='stop after this long (0 = until ctrl-c)')
    on.add_argument('--no-park', dest='park', action='store_false', help='yam: power down in place')
    on.add_argument('--viser', action='store_true')
    on.add_argument('--port', type=int, default=8090)
    on.add_argument('--log', type=Path)
    args = parser.parse_args()
    if args.mode == 'live' and args.min_z is None:
        args.min_z = args.table_z+.01
    raise SystemExit((offline if args.mode == 'offline' else live)(args) or 0)


if __name__ == '__main__':
    main()
