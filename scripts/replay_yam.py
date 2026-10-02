#!/usr/bin/env python3
"""Replay a raw yumi (T265 + D405) capture on a virtual YAM through karma's IK.

Run with karma's environment, which provides mujoco, viser and vr_teleop_kit:

    cd ~/karma && uv run python ../yumi/scripts/replay_yam.py ../yumi/data/<capture>

Targets go through EEFollower + DecoupledIKSolver, the same path a policy's
end-effector actions take on the robot. Nothing here talks to hardware.

With a world camera in the capture and configs/world_camera.json (from
calibrate_world.py), the arm, hand path and inner workspace are drawn over the
recorded overhead video. Align the base, then the UMI anchor, and save them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver
from vr_teleop_kit.ik.ee_follower import EEFollower

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from yumi.markers import Markers  # noqa: E402

READY = np.array([0., .7, .7, 0., 0., 0.])
# T265 pose frame (x right, y up, z backward) -> z-up, x-forward robot-style frame.
T265_TO_BASE = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
# Assumed mount until T_tracker_tcp is measured: T265 faces along the jaws, upright.
# Columns are YAM tool0 axes in the tracker frame: x (down) = -y, y (left) = -x,
# z (approach) = -z.
R_TRACKER_TCP_ASSUMED = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
YAM_FINGER_MAX = .0475  # joint7/joint8 travel; full opening is twice this
# Deployment velocity caps, matching karma's YAM teleop defaults (rad per IK tick).
MAX_DQ = [.06]*3+[.24]*3
POS_OK_MM, ROT_OK_DEG = 10., 5.


def wxyz(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def homogeneous(rotation, translation=(0., 0., 0.)):
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = rotation, translation
    return a


def load_capture(root, min_confidence, max_gap, rate):
    """Resample tracker poses onto a uniform grid; invalid where tracking is weak or gapped."""
    meta = json.loads((root/'metadata.json').read_text())
    rows = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    poses = [r for r in rows if r['kind'] == 'pose']
    images = [r for r in rows if r['kind'] == 'image']
    if len(poses) < 2:
        raise ValueError('The capture has no T265 pose stream.')
    if len({r['domain'] for r in poses+images}) != 1:
        raise ValueError('Pose and image timestamps are in different clock domains.')
    tp = np.array([r['device_s'] for r in poses])
    if np.any(np.diff(tp) <= 0):
        raise ValueError('Non-monotonic tracker timestamps.')
    confident = np.array([r['tracker_confidence'] >= min_confidence for r in poses])
    if not confident.any():
        raise ValueError(f'No tracker sample reaches confidence {min_confidence}.')
    t0 = tp[np.argmax(confident)]
    grid = np.arange(t0, tp[-1], 1/rate)
    position = np.array([r['position'] for r in poses])
    slerp = Slerp(tp, Rotation.from_quat([r['quaternion_xyzw'] for r in poses]))
    right = np.searchsorted(tp, grid, side='right').clip(1, len(tp)-1)
    left = right-1
    valid = confident[left] & confident[right] & (tp[right]-tp[left] <= max_gap)
    alpha = ((grid-tp[left])/(tp[right]-tp[left]))[:, None]
    world_tracker = np.tile(np.eye(4), (len(grid), 1, 1))
    world_tracker[:, :3, 3] = (1-alpha)*position[left]+alpha*position[right]
    world_tracker[:, :3, :3] = slerp(grid).as_matrix()
    return meta, images, grid-t0, grid, world_tracker, valid


def measure_widths(root, images, meta, marker_config):
    """Jaw opening per RGB frame (metres), None where markers fail the calibrated checks."""
    try:
        detector = Markers(marker_config, meta['rgb_intrinsics'])
    except (KeyError, ValueError) as exc:
        print(f'Width disabled: {exc}', flush=True)
        return [None]*len(images)

    def one(row):
        bgr = cv2.imread(str(root/row['rgb_path']))
        try:
            return float(detector.width(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[0])
        except (ValueError, cv2.error):
            return None
    with ThreadPoolExecutor() as pool:
        return list(pool.map(one, images))


def anchor_transform(anchor):
    """T265 world -> YAM base for an anchor (x, y, z m, yaw deg); gravity is shared."""
    x, y, z, yaw = anchor
    return homogeneous(Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ T265_TO_BASE, (x, y, z))


def start_anchor(world_tcp, valid, home):
    """Anchor placing the first valid TCP pose at the ready position, heading along robot +X."""
    first = homogeneous(T265_TO_BASE) @ world_tcp[np.argmax(valid)]
    yaw = -np.degrees(np.arctan2(first[1, 2], first[0, 2]))
    turned = anchor_transform((0., 0., 0., yaw)) @ world_tcp[np.argmax(valid)]
    return tuple(float(v) for v in home[:3, 3]-turned[:3, 3])+(float(yaw),)


def map_targets(world_tcp, valid, mode, home, anchor=None):
    """Hand TCP poses -> tool0 targets in the YAM base frame.

    relative: motion relative to the first valid TCP pose, applied in the tool frame
              from the ready pose (what a relative-action policy executes).
    gravity:  hand path rigidly placed at the ready position, keeping the hand's tilt
              against gravity and aligning its initial heading with robot +X.
    world:    one fixed T265 -> base anchor for the whole recording session, aligned
              against the world camera, so absolute workspace positions are kept.
    """
    if mode == 'relative':
        return home @ np.linalg.inv(world_tcp[np.argmax(valid)]) @ world_tcp
    if mode == 'gravity' or anchor is None:
        anchor = start_anchor(world_tcp, valid, home)
    return anchor_transform(anchor) @ world_tcp


def table_from_base(base):
    return homogeneous(Rotation.from_euler('z', base['yaw_deg'], degrees=True).as_matrix(),
                       (base['x_m'], base['y_m'], base['z_m']))


def project(points, cam_from_base, k):
    """Base-frame points -> pixel coordinates; rows behind the camera are dropped."""
    p = points @ cam_from_base[:3, :3].T+cam_from_base[:3, 3]
    p = p[p[:, 2] > .05]
    return np.c_[k['fx']*p[:, 0]/p[:, 2]+k['ppx'], k['fy']*p[:, 1]/p[:, 2]+k['ppy']]


def workspace(solver, base_height, samples=40000, cell=.02, voxel=.03):
    """Sampled reach of tool0: 2D table footprints for drawing, 3D voxels for checks.

    Inner excludes poses where the solver's shoulder damping ramp exceeds 0.5,
    i.e. near full extension or folding, where tracking degrades. Footprints cover
    tool0 positions up to 30 cm above the table.
    """
    rng = np.random.default_rng(0)
    limits = solver.joint_limits
    jacp = np.zeros((3, solver.model.nv))
    points, inner = [], []
    for q in rng.uniform(limits[:, 0], limits[:, 1], (samples, 6)):
        p, _ = solver.fk(q)
        if p[2]+base_height < 0:  # below the table
            continue
        mujoco.mj_jacSite(solver.model, solver.data, jacp, None, solver.j4_site_id)
        points.append(p)
        inner.append(abs(np.linalg.det(jacp[:, :3])) > .5*solver.w0)
    points, inner = np.array(points), np.array(inner)
    low = points[:, 2]+base_height <= .3
    origin = points[low, :2].min(0)-cell
    shape = tuple(((points[low, :2].max(0)-origin)/cell).astype(int)+2)
    grids = []
    for mask in (inner & low, low):
        grid = np.zeros(shape[::-1], np.uint8)
        ij = ((points[mask, :2]-origin)/cell).astype(int)
        grid[ij[:, 1], ij[:, 0]] = 255
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        grids.append(cv2.morphologyEx(grid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    cells = np.floor(points[inner]/voxel).astype(int)
    offsets = np.stack(np.meshgrid(*[[-1, 0, 1]]*3), -1).reshape(-1, 3)
    voxels = {tuple(c) for c in (cells[:, None]+offsets).reshape(-1, 3)}  # dilate one voxel
    return {'grids': grids, 'origin': origin, 'cell': cell, 'points': points[inner],
            'voxels': voxels, 'voxel': voxel}


def inside(space, points):
    return np.array([tuple(c) in space['voxels'] for c in np.floor(points/space['voxel']).astype(int)])


def pose_error(target, actual):
    return (float(np.linalg.norm(target[:3, 3]-actual[:3, 3])*1000),
            float(np.degrees(np.linalg.norm(Rotation.from_matrix(
                target[:3, :3] @ actual[:3, :3].T).as_rotvec()))))


def solve(capture, tracker_tcp, mode, rate, ik_hz, mu, anchor=None, inner=None):
    """Drive the deployment follower through every target; return per-sample results and a report."""
    meta, images, times, grid, world_tracker, valid, image_times, widths = capture
    solver = DecoupledIKSolver(mu=mu, max_dq_per_joint=MAX_DQ)
    follower = EEFollower(solver, freq=ik_hz, q_init=READY, gripper_max_speed=3.)
    pos, quat = solver.fk(READY)
    home = homogeneous(Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix(), pos)
    world_tcp = world_tracker @ tracker_tcp
    if mode == 'gravity' or (mode == 'world' and anchor is None):
        anchor = start_anchor(world_tcp, valid, home)
    targets = map_targets(world_tcp, valid, mode, home, anchor)
    nearest = np.abs(image_times[None, :]-grid[:, None]).argmin(1) if len(image_times) else None

    def command(i):
        width = widths[nearest[i]] if nearest is not None and \
            abs(image_times[nearest[i]]-grid[i]) < 1.5/rate else None
        gripper = None if width is None else float(np.clip(1-width/(2*YAM_FINGER_MAX), 0, 1))
        target = targets[i]
        follower.set_target(target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_quat(), gripper)
        return width

    def achieved():
        p, q = solver.fk(follower.qpos)
        return homogeneous(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), p)

    first = int(np.argmax(valid))
    command(first)
    for _ in range(int(3*ik_hz)):  # settle onto the start pose before t=0, as a rollout would
        follower.step()
    settle_error = pose_error(targets[first], achieved())
    substeps = max(1, round(ik_hz/rate))
    samples = []
    for i in range(len(grid)):
        width = command(i) if valid[i] else None
        singular = gimbal = pressure = 0.
        for _ in range(substeps):
            follower.step()
            singular = max(singular, solver.last_singularity_proximity)
            gimbal = max(gimbal, solver.last_wrist_gimbal_proximity)
            pressure = max(pressure, solver.last_limit_pressure)
        actual = achieved()
        err = pose_error(targets[i], actual) if valid[i] else (None, None)
        samples.append({'time': float(times[i]), 'valid': bool(valid[i]), 'q': follower.qpos,
            'gripper': follower.gripper, 'width': width, 'target': targets[i], 'actual': actual,
            'position_error_mm': err[0], 'rotation_error_deg': err[1], 'singularity': singular,
            'gimbal': gimbal, 'limit_pressure': pressure,
            'image': int(nearest[i]) if nearest is not None else None})
    flags = inner(targets[:, :3, 3]) if inner is not None else np.ones(len(samples), bool)
    for s, flag in zip(samples, flags):
        tracked = s['valid'] and s['position_error_mm'] <= POS_OK_MM and s['rotation_error_deg'] <= ROT_OK_DEG
        s['status'] = 'miss' if not tracked else 'ok' if flag else 'edge'
    good = [s for s in samples if s['valid']]
    pe = np.array([s['position_error_mm'] for s in good])
    re = np.array([s['rotation_error_deg'] for s in good])
    q = np.array([s['q'] for s in samples])
    limits = solver.joint_limits
    margin = np.minimum(q-limits[:, 0], limits[:, 1]-q).min(0)
    hand = targets[valid][:, :3, 3]
    measured = [w for w in widths if w is not None]
    tracked = (pe <= POS_OK_MM) & (re <= ROT_OK_DEG)
    report = {
        'mapping': mode,
        'anchor_xyz_m_yaw_deg': None if mode == 'relative' else [round(v, 4) for v in anchor],
        'T_tracker_tcp': tracker_tcp.round(6).tolist(),
        'samples': len(samples), 'valid_samples': len(good),
        'duration_s': float(times[-1]), 'rate_hz': rate, 'ik_hz': ik_hz, 'mu': mu,
        'start_settle_error': {'position_mm': settle_error[0], 'rotation_deg': settle_error[1]},
        'hand_path_span_m': (hand.max(0)-hand.min(0)).round(3).tolist(),
        'position_error_mm': {'median': float(np.median(pe)), 'p95': float(np.percentile(pe, 95)),
                              'max': float(pe.max())},
        'rotation_error_deg': {'median': float(np.median(re)), 'p95': float(np.percentile(re, 95)),
                               'max': float(re.max())},
        'tracked_fraction': float(tracked.mean()),
        'tracking_threshold': f'<= {POS_OK_MM:g} mm and <= {ROT_OK_DEG:g} deg',
        'joint_limit_margin_rad': margin.round(3).tolist(),
        'samples_pushing_limits': int(sum(s['limit_pressure'] > 1e-6 for s in samples)),
        'peak_joint_speed_rad_s': (np.abs(np.diff(q, axis=0)).max(0)*rate).round(2).tolist(),
        'max_shoulder_singularity_proximity': float(max(s['singularity'] for s in samples)),
        'max_wrist_gimbal_proximity': float(max(s['gimbal'] for s in samples)),
        'width': {'frames_measured': len(measured), 'frames': len(widths),
                  'min_m': min(measured, default=None), 'max_m': max(measured, default=None),
                  'beyond_yam_opening': int(sum(w > 2*YAM_FINGER_MAX for w in measured))},
        'hardware_connection': False,
    }
    if inner is not None:
        report['inner_workspace_fraction'] = float(inner(hand).mean())
    return samples, report


STATUS_COLORS = {'ok': (60, 190, 90), 'edge': (240, 170, 40), 'miss': (225, 60, 60)}
ARM_COLOR = (30, 160, 178)


def path_segments(samples, field, rate):
    pairs = [(a, b) for a, b in zip(samples, samples[1:]) if a['valid'] and b['valid']
             and b['time']-a['time'] <= 1.5/rate]
    points = np.asarray([[a[field][:3, 3], b[field][:3, 3]] for a, b in pairs], np.float32).reshape(-1, 2, 3)
    colors = np.asarray([[STATUS_COLORS[b['status']]]*2 for _, b in pairs], np.uint8).reshape(-1, 2, 3)
    return points, colors


def tracker_tcp_from(rotation, offset, tweak_deg):
    return homogeneous(rotation @ Rotation.from_euler('xyz', tweak_deg, degrees=True).as_matrix(),
                       offset)


def summary_markdown(report):
    pe, re = report['position_error_mm'], report['rotation_error_deg']
    w = report['width']
    workspace_line = (f"\n\nInside inner workspace **{report['inner_workspace_fraction']*100:.0f}%**"
                      if 'inner_workspace_fraction' in report else '')
    width = (f"{w['min_m']*1000:.0f}–{w['max_m']*1000:.0f} mm on {w['frames_measured']}/{w['frames']} frames"
             if w['frames_measured'] else 'not measured')
    return (f"**{report['tracked_fraction']*100:.0f}%** of samples tracked "
            f"({report['tracking_threshold']})\n\n"
            f"Position: median {pe['median']:.1f} · p95 {pe['p95']:.1f} · max {pe['max']:.1f} mm\n\n"
            f"Rotation: median {re['median']:.1f} · p95 {re['p95']:.1f} · max {re['max']:.1f}°\n\n"
            f"Hand span {report['hand_path_span_m']} m · start settle "
            f"{report['start_settle_error']['position_mm']:.1f} mm / "
            f"{report['start_settle_error']['rotation_deg']:.1f}°\n\n"
            f"Joint margin (rad) {report['joint_limit_margin_rad']}\n\n"
            f"Peak joint speed (rad/s) {report['peak_joint_speed_rad_s']}\n\n"
            f"Max gimbal proximity {report['max_wrist_gimbal_proximity']:.2f} · "
            f"shoulder {report['max_shoulder_singularity_proximity']:.2f}\n\n"
            f"Jaw width {width}"+workspace_line)


def serve(args, capture, state, tracker_rotation, world_calib, space):
    import viser

    meta, images, grid = capture[0], capture[1], capture[3]
    world = [r for r in map(json.loads, (args.capture/'samples.jsonl').read_text().splitlines())
             if r['kind'] == 'world']
    world_times = np.array([r['device_s'] for r in world])
    aligned = bool(world and world_calib)
    server = viser.ViserServer(host=args.host, port=args.port, label='yumi · YAM replay')
    server.scene.set_up_direction('+z')
    server.gui.configure_theme(control_width='medium',
                               brand_color=(39, 135, 173), show_share_button=False)
    floor = server.scene.add_grid('/floor', width=2.5, height=2.5, cell_size=.1, section_size=.5,
                                  plane_color=(239, 243, 247), plane_opacity=1, position=(0, 0, -.004))
    server.scene.add_frame('/base', axes_length=.12, axes_radius=.003)
    solver = DecoupledIKSolver()
    model, data = solver.model, solver.data
    handles = {}
    arm_points = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        color = (45, 55, 70) if gid in (0, 3, 6) else (180, 192, 202)
        if gid >= model.ngeom-2:
            color = (52, 120, 151)
        handles[gid] = server.scene.add_mesh_simple(f'/yam/geom_{gid}',
            vertices=model.mesh_vert[va:va+vn].astype(np.float32),
            faces=model.mesh_face[fa:fa+fn].astype(np.uint32), color=color)
        arm_points[gid] = model.mesh_vert[va:va+vn][::max(1, vn//400)]
    target_path = server.scene.add_line_segments('/requested_path', points=np.zeros((1, 2, 3), np.float32),
                                                 colors=(243, 151, 50), line_width=3)
    arm_path = server.scene.add_line_segments('/arm_path', points=np.zeros((1, 2, 3), np.float32),
                                              colors=ARM_COLOR, line_width=1.5)
    target_frame = server.scene.add_frame('/requested', axes_length=.09, axes_radius=.002)
    actual_frame = server.scene.add_frame('/actual', axes_length=.065, axes_radius=.002)
    error_line = server.scene.add_line_segments('/error', points=np.zeros((1, 2, 3), np.float32),
                                                colors=(230, 90, 70), line_width=3)
    duration = state['report']['duration_s']
    server.gui.add_markdown(f'## YAM · UMI capture\n**{args.capture.name}**\n\n'
        f'{duration:.1f} s · T265 + D405 · {state["report"]["samples"]:,} samples @ {args.rate:g} Hz',
        order=0)
    playing = server.gui.add_checkbox('Play', True, order=1)
    timeline = server.gui.add_slider('Time (s)', min=0., max=duration, step=.01, initial_value=0., order=1.1)
    status = server.gui.add_markdown('Loading…', order=3)
    server.gui.add_markdown('Hand path: 🟢 reached, inner workspace · 🟠 reached, edge of reach · '
                            '🔴 missed (>10 mm or >5°). Teal: arm tool0.', order=3.1)
    blank = np.full((320, 480, 3), (27, 34, 43), np.uint8)
    with server.gui.add_folder('Recorded D405 camera', order=2):
        camera_image = server.gui.add_image(blank, label='D405 RGB', format='jpeg', jpeg_quality=85)
        camera_rotation = server.gui.add_dropdown('Rotate view',
            options=('0°', '90° clockwise', '180°', '90° counterclockwise'), initial_value='0°')
    if world:
        with server.gui.add_folder('Recorded world camera', order=2.5):
            world_image = server.gui.add_image(blank, label=meta['world_camera']['name'],
                                               format='jpeg', jpeg_quality=85)
            if aligned:
                show_overlay = server.gui.add_checkbox('Overlay', True)
                show_reach = server.gui.add_checkbox('Full-reach outline on table', False)
                park = server.gui.add_checkbox('Arm at park pose (q = 0) for base alignment', False)
    if aligned:
        base = world_calib['base']
        with server.gui.add_folder('World camera alignment', order=6, expand_by_default=False):
            fit = world_calib['table_fit']
            server.gui.add_markdown(f'Camera {fit.get("measured_camera_height_m", fit["camera_height_m"])*100:.0f} cm '
                                    f'above the table (measured), {fit["tilt_from_vertical_deg"]:.1f}° tilt '
                                    f'(depth fit). Align the virtual {base.get("arm", "")} arm with the real one, '
                                    'then the UMI anchor so the orange dot sits on the gripper.')
            base_sliders = [server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=float(base[key]))
                for label, key, lo, hi, step in (('Base x (m, table)', 'x_m', -.8, .8, .005),
                                                 ('Base y (m, table)', 'y_m', -.8, .8, .005),
                                                 ('Base z above table (m)', 'z_m', -.1, .3, .005),
                                                 ('Base yaw (°)', 'yaw_deg', -180., 180., .5))]
            anchor = state['report'].get('anchor_xyz_m_yaw_deg') or (0., 0., 0., 0.)
            anchor_sliders = [server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=float(v))
                for (label, lo, hi, step), v in zip((('UMI anchor x (m, base)', -1.5, 1.5, .005),
                                                     ('UMI anchor y (m, base)', -1.5, 1.5, .005),
                                                     ('UMI anchor z (m, base)', -1.5, 1.5, .005),
                                                     ('UMI anchor yaw (°)', -180., 180., .5)), anchor)]
            server.gui.add_markdown('Anchor sliders apply in the **world** mapping after Re-solve IK.')
            save = server.gui.add_button('Save alignment')
            saved = server.gui.add_markdown('')
        table_cam = np.asarray(world_calib['T_table_cam'])
        k = world_calib['intrinsics']
        frustum = server.scene.add_camera_frustum('/world_camera', fov=2*np.arctan(k['height']/2/k['fy']),
            aspect=k['width']/k['height'], scale=.12, color=(120, 120, 120))
        contours = [[c.reshape(-1, 2)*space['cell']+space['origin']+space['cell']/2
                     for c in cv2.findContours(g, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]]
                    for g in space['grids']]

        def base_values():
            return dict(zip(('x_m', 'y_m', 'z_m', 'yaw_deg'), (b.value for b in base_sliders)))

        def cam_from_base():
            return np.linalg.inv(np.linalg.inv(table_from_base(base_values())) @ table_cam)

        @server.gui.add_button('View from world camera', order=1.2).on_click
        def _(event):
            pose = np.linalg.inv(cam_from_base())
            event.client.camera.position = pose[:3, 3]
            event.client.camera.look_at = pose[:3, 3]+.8*pose[:3, 2]
            event.client.camera.up_direction = -pose[:3, 1]
            event.client.camera.fov = 2*np.arctan(k['height']/2/k['fy'])

        @save.on_click
        def _(_):
            world_calib['base'] = {'arm': base.get('arm', 'right'), **base_values()}
            args.world_calib.write_text(json.dumps(world_calib, indent=2)+'\n')
            session = {'anchor_xyz_m_yaw_deg': [a.value for a in anchor_sliders],
                       'mount_tweak_deg': [t.value for t in tweaks],
                       'tcp_offset_m': [o.value/100 for o in offsets]}
            path = args.capture.parent/'world_anchor.json'
            path.write_text(json.dumps(session, indent=2)+'\n')
            saved.content = f'Saved base → `{args.world_calib.name}`, anchor and mount → `{path}`'
    with server.gui.add_folder('Playback options', order=4, expand_by_default=False):
        loop = server.gui.add_checkbox('Loop', True)
        speed = server.gui.add_slider('Playback speed', min=.25, max=2., step=.25, initial_value=1.)
        restart = server.gui.add_button('Restart')
        show_path = server.gui.add_checkbox('Show paths', True)
    with server.gui.add_folder('UMI → YAM mapping', order=5):
        server.gui.add_markdown(f'T_tracker_tcp: **{state["tcp_source"]}**. Tweak the mount and '
                                'offset, then re-solve. Orange: hand target. Teal: arm tool0.')
        mode = server.gui.add_dropdown('Mapping', options=('relative', 'gravity', 'world'),
                                       initial_value=args.mapping)
        tweaks = [server.gui.add_slider(f'Mount {axis} (°, tool axes)', min=-180., max=180., step=1.,
                                        initial_value=float(v)) for axis, v in zip('xyz', args.mount_tweak)]
        offsets = [server.gui.add_slider(f'TCP offset {axis} (cm, tracker frame)', min=-30., max=30.,
                   step=.5, initial_value=float(v*100)) for axis, v in zip('xyz', args.tcp_offset)]
        resolve = server.gui.add_button('Re-solve IK')
        summary = server.gui.add_markdown(summary_markdown(state['report']))
    commands = queue.SimpleQueue()
    timeline.on_update(lambda event: commands.put(('seek', timeline.value)) if event.client is not None else None)
    restart.on_click(lambda _: commands.put(('seek', 0.)))
    redraw = threading.Event()
    redraw.set()

    @resolve.on_click
    def _(_):
        resolve.disabled = True
        summary.content = 'Solving…'

        def work():
            tracker_tcp = tracker_tcp_from(tracker_rotation, np.array([o.value for o in offsets])/100,
                                           [t.value for t in tweaks])
            anchor = [a.value for a in anchor_sliders] if aligned and mode.value == 'world' else None
            samples, report = solve(capture, tracker_tcp, mode.value, args.rate, args.ik_hz, args.mu,
                                    anchor, state.get('inner'))
            state.update(samples=samples, report=report)
            if aligned and report['anchor_xyz_m_yaw_deg']:
                for slider, v in zip(anchor_sliders, report['anchor_xyz_m_yaw_deg']):
                    slider.value = float(v)
            summary.content = summary_markdown(report)
            print(json.dumps({k: report[k] for k in ('mapping', 'T_tracker_tcp', 'tracked_fraction',
                  'position_error_mm', 'rotation_error_deg')}), flush=True)
            resolve.disabled = False
            redraw.set()
        threading.Thread(target=work, daemon=True).start()

    @server.on_client_connect
    def connected(client):
        client.camera.position = (1.15, -1.15, .95)
        client.camera.look_at = (.25, 0, .3)
        client.camera.up_direction = (0, 0, 1)

    print(f'YAM replay: http://127.0.0.1:{args.port} (Ctrl-C to stop)', flush=True)
    position, previous, last_index, last_camera = 0., time.monotonic(), -1, None
    last_world = None

    def draw_world(rgb, sample, samples):
        image = rgb.copy()
        cam, height = cam_from_base(), base_sliders[2].value
        def outline(xy):
            return project(np.c_[xy, np.full(len(xy), -height)], cam, k).astype(np.int32)
        if show_reach.value:
            cv2.polylines(image, [outline(xy) for xy in contours[1]], True, (150, 150, 150), 1, cv2.LINE_AA)
        data.qpos[:] = 0
        if not park.value:
            data.qpos[:6] = sample['q']
            data.qpos[6:8] = (1-sample['gripper'])*YAM_FINGER_MAX
        mujoco.mj_forward(model, data)
        for gid, v in arm_points.items():
            for u in project(v @ data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid], cam, k).astype(int):
                cv2.circle(image, tuple(u), 1, (70, 120, 230), -1)
        segments, colors = path_segments(samples, 'target', args.rate)
        for (a, b), (color, _) in zip(segments, colors):
            uv = project(np.array([a, b], float), cam, k)
            if len(uv) == 2:
                cv2.line(image, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)),
                         tuple(int(c) for c in color), 2, cv2.LINE_AA)
        for field, radius in (('target', 6), ('actual', 4)):
            dot = project(sample[field][None, :3, 3], cam, k)
            color = STATUS_COLORS[sample['status']] if field == 'target' else ARM_COLOR
            if len(dot):
                cv2.circle(image, tuple(dot[0].astype(int)), radius, color, -1 if field == 'target' else 2)
        return image
    rotations = {'0°': None, '90° clockwise': cv2.ROTATE_90_CLOCKWISE, '180°': cv2.ROTATE_180,
                 '90° counterclockwise': cv2.ROTATE_90_COUNTERCLOCKWISE}
    try:
        while True:
            now = time.monotonic()
            elapsed, previous = min(now-previous, .1), now
            if playing.value and server.get_clients():
                position += elapsed*speed.value
            while not commands.empty():
                _, position = commands.get()
                last_index = -1
            if position > duration:
                position = position % duration if loop.value else duration
                playing.value = playing.value and loop.value
            samples = state['samples']
            index = min(len(samples)-1, int(position*args.rate))
            sample = samples[index]
            with server.atomic():
                if redraw.is_set():
                    target_path.points, target_path.colors = path_segments(samples, 'target', args.rate)
                    arm_path.points = path_segments(samples, 'actual', args.rate)[0]
                    redraw.clear()
                    last_index = -1
                target_path.visible = arm_path.visible = show_path.value
                if index != last_index:
                    data.qpos[:6] = sample['q']
                    data.qpos[6:8] = (1-sample['gripper'])*YAM_FINGER_MAX
                    mujoco.mj_forward(model, data)
                    for gid, handle in handles.items():
                        handle.position = data.geom_xpos[gid].copy()
                        handle.wxyz = wxyz(data.geom_xmat[gid].reshape(3, 3))
                    actual_frame.position = sample['actual'][:3, 3]
                    actual_frame.wxyz = wxyz(sample['actual'][:3, :3])
                    target_frame.position = sample['target'][:3, 3]
                    target_frame.wxyz = wxyz(sample['target'][:3, :3])
                    error_line.points = np.array([[sample['target'][:3, 3], sample['actual'][:3, 3]]], np.float32)
                    last_index = index
                target_frame.visible = error_line.visible = sample['valid']
                timeline.value = float(position)
                if sample['valid']:
                    width = 'n/a' if sample['width'] is None else f'{sample["width"]*1000:.0f} mm'
                    flags = [name for name, v in (('gimbal', sample['gimbal']), ('shoulder', sample['singularity']))
                             if v > .5] + (['joint limit'] if sample['limit_pressure'] > 1e-6 else [])
                    status.content = (f'**{position:.2f} / {duration:.2f} s** · '
                        f'error **{sample["position_error_mm"]:.1f} mm** / '
                        f'**{sample["rotation_error_deg"]:.1f}°**\n\n'
                        f'Jaw {width}' + (f' · ⚠ {", ".join(flags)}' if flags else ''))
                else:
                    status.content = f'**{position:.2f} s** · tracker confidence low or gap — arm holds'
                key = (sample['image'], camera_rotation.value)
                if sample['image'] is not None and key != last_camera:
                    rgb = cv2.cvtColor(cv2.imread(str(args.capture/images[sample['image']]['rgb_path'])),
                                       cv2.COLOR_BGR2RGB)
                    if rotations[camera_rotation.value] is not None:
                        rgb = cv2.rotate(rgb, rotations[camera_rotation.value])
                    camera_image.image = rgb
                    last_camera = key
                if world:
                    w = int(np.abs(world_times-grid[index]).argmin())
                    key = (w, index, id(samples)) + ((show_overlay.value, park.value, show_reach.value,
                           *(b.value for b in base_sliders)) if aligned else ())
                    if key != last_world:
                        rgb = cv2.cvtColor(cv2.imread(str(args.capture/world[w]['rgb_path'])), cv2.COLOR_BGR2RGB)
                        if aligned:
                            base_cam = np.linalg.inv(cam_from_base())
                            frustum.position = base_cam[:3, 3]
                            frustum.wxyz = wxyz(base_cam[:3, :3])
                            floor.position = (0, 0, -base_sliders[2].value-.004)
                            if show_overlay.value:
                                rgb = draw_world(rgb, sample, samples)
                        world_image.image = rgb
                        last_world = key
            time.sleep(1/30)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('capture', type=Path, help='Raw yumi capture directory (samples.jsonl, metadata.json)')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/umi.local.json',
                        help='yumi config with T_tracker_tcp and marker width calibration')
    parser.add_argument('--mapping', choices=('relative', 'gravity', 'world'), default='gravity',
                        help='world uses the saved session anchor (world_anchor.json) when present')
    parser.add_argument('--world-calib', type=Path, default=ROOT/'configs/world_camera.json',
                        help='World camera calibration from scripts/calibrate_world.py')
    parser.add_argument('--tcp-offset', type=float, nargs=3, default=(0., 0., 0.), metavar=('X', 'Y', 'Z'),
                        help='Tracker->TCP translation in the tracker frame (m) when T_tracker_tcp is unset')
    parser.add_argument('--mount-tweak', type=float, nargs=3, default=(0., 0., 0.), metavar=('RX', 'RY', 'RZ'),
                        help='Extra mount rotation about the TCP axes (deg)')
    parser.add_argument('--rate', type=float, default=30., help='Target rate, like a policy action stream (Hz)')
    parser.add_argument('--ik-hz', type=float, default=200., help='EEFollower inner loop rate (Hz)')
    parser.add_argument('--mu', type=float, default=.005, help='Posture bias; EEFollower deploy maximum')
    parser.add_argument('--min-confidence', type=int, choices=(1, 2, 3),
                        help='Override the config tracker-confidence gate (preview only below 3)')
    parser.add_argument('--no-width', action='store_true', help='Skip marker width detection')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--check', action='store_true', help='Solve and report without serving')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    min_confidence = args.min_confidence or config.get('min_tracker_confidence', 3)
    meta, images, times, grid, world_tracker, valid = load_capture(
        args.capture, min_confidence, config.get('max_pose_gap_s', .025), args.rate)
    if config.get('T_tracker_tcp') is not None:
        measured = np.asarray(config['T_tracker_tcp'], dtype=float)
        tracker_rotation, args.tcp_offset, tcp_source = measured[:3, :3], measured[:3, 3], 'measured (config)'
    else:
        tracker_rotation, tcp_source = R_TRACKER_TCP_ASSUMED, 'ASSUMED — not calibrated'
    image_times = np.array([r['device_s'] for r in images])+config.get('camera_time_offset_s', 0.)
    widths = ([None]*len(images) if args.no_width
              else measure_widths(args.capture, images, meta, config.get('markers', meta['config']['markers'])))
    capture = (meta, images, times, grid, world_tracker, valid, image_times, widths)
    world_calib = json.loads(args.world_calib.read_text()) if args.world_calib.exists() else None
    space, inner, anchor = None, None, None
    if world_calib:
        space = workspace(DecoupledIKSolver(), world_calib['base']['z_m'])
        inner = lambda points: inside(space, points)  # noqa: E731
    session = args.capture.parent/'world_anchor.json'
    if session.exists():
        saved = json.loads(session.read_text())
        anchor = saved['anchor_xyz_m_yaw_deg']
        if config.get('T_tracker_tcp') is None:
            args.mount_tweak, args.tcp_offset = saved['mount_tweak_deg'], saved['tcp_offset_m']
            tcp_source += f' · mount tweak {args.mount_tweak}° from {session.name}'
    tracker_tcp = tracker_tcp_from(tracker_rotation, np.asarray(args.tcp_offset), args.mount_tweak)
    samples, report = solve(capture, tracker_tcp, args.mapping, args.rate, args.ik_hz, args.mu, anchor, inner)
    report = {'capture': str(args.capture.resolve()), 'T_tracker_tcp_source': tcp_source,
              'min_tracker_confidence': min_confidence, **report}
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    if not args.check:
        serve(args, capture, {'samples': samples, 'report': report, 'tcp_source': tcp_source,
                              'inner': inner}, tracker_rotation, world_calib, space)


if __name__ == '__main__':
    main()
