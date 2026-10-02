#!/usr/bin/env python3
"""Convert raw yumi (T265 + D405) episodes into a UMI-style LeRobot v3 dataset.

Run with karma's environment (LeRobot 0.6.1):

    cd ~/karma && uv run python ../yumi/scripts/export_umi_lerobot.py \\
        ../yumi/data/raw/session-02 --repo-id vruga/yumi-umi-block-place \\
        --task "pick up the green block and place it on the wooden board" --push

Observation and action design follows the UMI policy interface:

* Latency matching (PD1.1): the wrist RGB frame is the timing reference; the
  T265 pose and the marker jaw width are interpolated to each image's capture
  time (plus camera_time_offset_s from the config, currently uncalibrated).
* Relative EE poses (PD2): every pose is expressed in the episode-initial TCP
  frame, rotations as 6D (first two rotation-matrix columns).
* Observation horizon 2 (PD2.2): observation.state is the pose `--obs-step`
  frames ago relative to the current pose (velocity information), plus both
  jaw widths. The current pose relative to itself is identity and omitted.
* Relative trajectory actions (PD2.1): `action` stores the next pose in the
  episode frame, the usual LeRobot convention. Train with an action chunk via
  delta_timestamps and call `relative_action_chunk` to express the chunk
  relative to the current pose, as UMI's own dataloader does.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from yumi.markers import Markers  # noqa: E402

# Assumed until T_tracker_tcp is measured; same convention as replay_yam.py.
R_TRACKER_TCP_ASSUMED = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
POSE_NAMES = ['x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21']


def rot6d(rotation):
    return rotation[..., :, :2].swapaxes(-1, -2).reshape(*rotation.shape[:-2], 6)


def rot6d_to_matrix(d6):
    """Inverse of rot6d via Gram-Schmidt; works on (..., 6) arrays or tensors."""
    a, b = d6[..., :3], d6[..., 3:6]
    x = a/np.linalg.norm(a, axis=-1, keepdims=True)
    b = b-(x*b).sum(-1, keepdims=True)*x
    y = b/np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def pose10(T, width):
    return np.concatenate([T[..., :3, 3], rot6d(T[..., :3, :3]), np.asarray(width, float)[..., None]], -1)


def relative_action_chunk(current_pose10, action_chunk10):
    """UMI relative trajectory: each chunk pose relative to the current EE pose.

    current_pose10: (..., 10) observation.ee_pose at t.
    action_chunk10: (..., H, 10) actions at t..t+H-1 from delta_timestamps.
    Returns (..., H, 10): relative position, relative rot6d, absolute width.
    """
    def matrix(p):
        T = np.zeros(p.shape[:-1]+(4, 4))
        T[..., :3, :3] = rot6d_to_matrix(p[..., 3:9])
        T[..., :3, 3] = p[..., :3]
        T[..., 3, 3] = 1
        return T
    current, chunk = matrix(np.asarray(current_pose10)[..., None, :]), matrix(np.asarray(action_chunk10))
    relative = np.linalg.inv(current) @ chunk
    return np.concatenate([relative[..., :3, 3], rot6d(relative[..., :3, :3]),
                           np.asarray(action_chunk10)[..., 9:10]], -1)


def width_model(samples_path, exclude_mm):
    d = json.loads(Path(samples_path).read_text())
    px, mm = np.array(d['marker_separation_px']), np.array(d['jaw_opening_m'])*1000
    keep = ~np.isin(np.round(mm, 3), np.round(exclude_mm, 3))
    gain, offset = np.polyfit(px[keep], mm[keep], 1)
    residual = mm[keep]-(gain*px[keep]+offset)
    return {'gain_mm_per_px': float(gain), 'offset_mm': float(offset),
            'fit_max_residual_mm': float(np.abs(residual).max()),
            'fit_rms_mm': float(np.sqrt((residual**2).mean())),
            'separation_range_px': [float(px[keep].min()), float(px[keep].max())],
            'samples_used': int(keep.sum()), 'excluded_openings_mm': list(exclude_mm),
            'source': str(samples_path)}


def load_episode(root, config, widths_cal, tracker_tcp, min_confidence, max_gap, with_world):
    meta = json.loads((root/'metadata.json').read_text())
    rows = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    images = [r for r in rows if r['kind'] == 'image']
    poses = [r for r in rows if r['kind'] == 'pose']
    tp = np.array([r['device_s'] for r in poses])
    ti = np.array([r['device_s'] for r in images])+config.get('camera_time_offset_s', 0.)
    if np.any(np.diff(tp) <= 0) or np.any(np.diff(ti) <= 0):
        raise ValueError('non-monotonic timestamps')
    confident = np.array([r['tracker_confidence'] >= min_confidence for r in poses])
    right = np.searchsorted(tp, ti, side='right')
    inside = (right > 0) & (right < len(tp))
    right = right.clip(1, len(tp)-1)
    left = right-1
    ok = inside & confident[left] & confident[right] & (tp[right]-tp[left] <= max_gap)
    if not ok.any():
        raise ValueError('no frame with confident, bracketed tracking')
    first, last = np.argmax(ok), len(ok)-1-np.argmax(ok[::-1])
    if not ok[first:last+1].all():
        raise ValueError(f'tracking gap or low confidence inside the episode ({(~ok[first:last+1]).sum()} frames)')
    keep = np.arange(first, last+1)
    alpha = ((ti[keep]-tp[left[keep]])/(tp[right[keep]]-tp[left[keep]]))[:, None]
    position = np.array([r['position'] for r in poses])
    world_tracker = np.tile(np.eye(4), (len(keep), 1, 1))
    world_tracker[:, :3, 3] = (1-alpha)*position[left[keep]]+alpha*position[right[keep]]
    world_tracker[:, :3, :3] = Slerp(tp, Rotation.from_quat([r['quaternion_xyzw'] for r in poses]))(
        ti[keep]).as_matrix()
    world_tcp = world_tracker @ tracker_tcp
    episode_tcp = np.linalg.inv(world_tcp[0]) @ world_tcp

    detector = Markers(config['markers'], meta['rgb_intrinsics'])

    def separation(row):
        rgb = cv2.cvtColor(cv2.imread(str(root/row['rgb_path'])), cv2.COLOR_BGR2RGB)
        try:
            return detector.image_separation(rgb)[0]
        except ValueError:
            return np.nan
    with ThreadPoolExecutor() as pool:
        px = np.array(list(pool.map(separation, [images[i] for i in keep])))
    measured = np.isfinite(px)
    if not measured.any():
        raise ValueError('jaw markers never detected')
    mm = widths_cal['gain_mm_per_px']*px+widths_cal['offset_mm']
    t = ti[keep]
    width = np.interp(t, t[measured], mm[measured])/1000  # hold at the ends, linear in gaps
    gaps = np.diff(np.flatnonzero(np.r_[True, measured, True]))-1
    lo, hi = widths_cal['separation_range_px']
    world = None
    if with_world:
        world_rows = [r for r in rows if r['kind'] == 'world']
        tw = np.array([r['device_s'] for r in world_rows])
        world = [world_rows[int(np.abs(tw-x).argmin())] for x in t]
    return {'root': root, 'times': t-t[0], 'pose': episode_tcp, 'width': width, 'measured': measured,
            'images': [images[i] for i in keep], 'world': world,
            'stats': {'frames': len(keep), 'trimmed_frames': len(images)-len(keep),
                      'width_measured_fraction': float(measured.mean()),
                      'longest_width_gap_frames': int(gaps.max()),
                      'width_extrapolated_fraction': float(((px < lo) | (px > hi))[measured].mean()),
                      'width_range_mm': [float(width.min()*1000), float(width.max()*1000)]}}


def frames(ep, obs_step):
    pose, width = ep['pose'], ep['width']
    n = len(pose)
    for i in range(n):
        j = max(0, i-obs_step)  # UMI pads the history with the first frame
        previous = np.linalg.inv(pose[i]) @ pose[j]
        nxt = min(i+1, n-1)
        yield i, {
            'observation.state': np.r_[pose10(previous, width[j])[:9], width[j], width[i]].astype(np.float32),
            'observation.ee_pose': pose10(pose[i], width[i]).astype(np.float32),
            'observation.gripper_width_measured': np.array([float(ep['measured'][i])], np.float32),
            'action': pose10(pose[nxt], width[nxt]).astype(np.float32),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session', type=Path, help='Session directory with episode-NNN folders')
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--task', required=True, help='Language instruction stored with every frame')
    parser.add_argument('--root', type=Path, help='Output folder (default data/lerobot/<repo name>)')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/umi.local.json')
    parser.add_argument('--width-samples', type=Path, default=ROOT/'data/calibration/width-2026-10-02.json')
    parser.add_argument('--exclude-width-mm', type=float, nargs='*', default=[48.0],
                        help='Calibration openings to drop (48 mm was non-monotonic)')
    parser.add_argument('--skip', nargs='*', default=[], help='Episode names to leave out, e.g. episode-000')
    parser.add_argument('--min-seconds', type=float, default=4.)
    parser.add_argument('--obs-step', type=int, default=3,
                        help='Frames between the two observation steps (3 at 30 fps = 10 Hz history)')
    parser.add_argument('--with-world', action='store_true', help='Also store the overhead camera video')
    parser.add_argument('--push', action='store_true', help='Upload to the Hugging Face Hub')
    parser.add_argument('--public', action='store_true', help='Make the Hub dataset public (default private)')
    args = parser.parse_args()

    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = json.loads(args.config.read_text())
    cal = width_model(args.width_samples, args.exclude_width_mm)
    tracker_tcp = np.eye(4)
    tracker_tcp[:3, :3] = R_TRACKER_TCP_ASSUMED
    tcp_source = 'assumed (T265 facing along the jaws, upright, zero offset)'
    if config.get('T_tracker_tcp') is not None:
        tracker_tcp, tcp_source = np.asarray(config['T_tracker_tcp'], float), 'measured (config)'
    root = args.root or ROOT/'data/lerobot'/args.repo_id.split('/')[-1]
    if root.exists():
        raise FileExistsError(f'{root} exists; remove it or pass --root')

    episodes, skipped = [], {}
    for path in sorted(args.session.glob('episode-*')):
        if path.name in args.skip:
            skipped[path.name] = 'skipped by request'
        elif not (path/'COMPLETE').exists() or (path/'FAILED.txt').exists():
            skipped[path.name] = 'incomplete capture'
        else:
            try:
                ep = load_episode(path, config, cal, tracker_tcp, config.get('min_tracker_confidence', 3),
                                  config.get('max_pose_gap_s', .025), args.with_world)
            except ValueError as exc:
                skipped[path.name] = str(exc)
                continue
            if ep['times'][-1] < args.min_seconds:
                skipped[path.name] = f'shorter than {args.min_seconds:g} s'
                continue
            episodes.append(ep)
            print(f'{path.name}: {json.dumps(ep["stats"])}', flush=True)
    for name, reason in skipped.items():
        print(f'{name}: SKIPPED ({reason})', flush=True)
    if not episodes:
        raise SystemExit('No episodes to export.')

    fps = config['fps']
    h, w = cv2.imread(str(episodes[0]['root']/episodes[0]['images'][0]['rgb_path'])).shape[:2]
    video = {'dtype': 'video', 'shape': (h, w, 3), 'names': ['height', 'width', 'channels']}
    features = {
        'observation.images.wrist': video,
        'observation.state': {'dtype': 'float32', 'shape': (11,),
            'names': [f'prev_rel_{n}' for n in POSE_NAMES]+['prev_width', 'width']},
        'observation.ee_pose': {'dtype': 'float32', 'shape': (10,), 'names': POSE_NAMES+['width']},
        'observation.gripper_width_measured': {'dtype': 'float32', 'shape': (1,), 'names': ['measured']},
        'action': {'dtype': 'float32', 'shape': (10,), 'names': POSE_NAMES+['width']},
    }
    if args.with_world:
        features['observation.images.world'] = video
    os.environ.setdefault('SVT_LOG', '1')
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, root=root, robot_type='umi_yam',
                                    features=features, use_videos=True,
                                    rgb_encoder=RGBEncoderConfig(preset=10))
    for number, ep in enumerate(episodes):
        for i, frame in frames(ep, args.obs_step):
            frame['observation.images.wrist'] = cv2.cvtColor(
                cv2.imread(str(ep['root']/ep['images'][i]['rgb_path'])), cv2.COLOR_BGR2RGB)
            if args.with_world:
                frame['observation.images.world'] = cv2.cvtColor(
                    cv2.imread(str(ep['root']/ep['world'][i]['rgb_path'])), cv2.COLOR_BGR2RGB)
            frame['task'] = args.task
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f'saved {number+1}/{len(episodes)} ({ep["root"].name})', flush=True)
    dataset.finalize()

    export = {
        'source_session': str(args.session.resolve()),
        'episodes': [ep['root'].name for ep in episodes], 'skipped': skipped,
        'per_episode': {ep['root'].name: ep['stats'] for ep in episodes},
        'fps': fps, 'obs_step_frames': args.obs_step,
        'T_tracker_tcp': tracker_tcp.round(6).tolist(), 'T_tracker_tcp_source': tcp_source,
        'camera_time_offset_s': config.get('camera_time_offset_s', 0.),
        'width_calibration': cal,
        'semantics': {
            'observation.state': 'pose obs_step frames ago relative to the current TCP pose '
                                 '(pos m, rot6d) + previous and current jaw width (m)',
            'observation.ee_pose': 'current TCP pose in the episode-initial TCP frame + width',
            'action': 'next TCP pose in the episode-initial TCP frame + width; convert chunks with '
                      'relative_action_chunk for UMI relative-trajectory actions',
            'rot6d': 'first two columns of the rotation matrix',
            'width': 'jaw opening (m) from marker separation, linearly interpolated over undetected frames'},
    }
    (root/'meta'/'umi_export.json').write_text(json.dumps(export, indent=2)+'\n')
    shutil.copy(__file__, root/'meta'/'export_umi_lerobot.py')

    check = LeRobotDataset(args.repo_id, root=root)
    sample = check[len(check)//2]
    print(json.dumps({'root': str(root), 'episodes': check.num_episodes, 'frames': check.num_frames,
                      'keys': sorted(k for k in sample if not k.startswith('observation.images')),
                      'image_shape': list(sample['observation.images.wrist'].shape)}, indent=2), flush=True)
    if args.push:
        dataset.push_to_hub(private=not args.public, tags=['umi', 'yam', 't265', 'd405'])
        print(f'Pushed: https://huggingface.co/datasets/{args.repo_id}', flush=True)


if __name__ == '__main__':
    main()
