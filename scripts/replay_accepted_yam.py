#!/usr/bin/env python3
"""Replay an accepted UMI trajectory through Karma's right-arm lifecycle.

Default is an offline Viser preview. --execute explicitly enables hardware.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from yam_motion import JointStreamer, Planner, PoseIK
from umi_review_state import ROOT, fingerprint

DEFAULT_MANIFEST = ROOT/'data/umi-runs/accepted/20261011-012155-5426da--episode-001.json'
HZ = 200
VMAX = np.array([1.5]*3+[3.]*3)
AMAX = np.array([6.]*3+[12.]*3)


def simulate(q0, open0, times, qs, openings):
    """The existing smooth stream, including convergence to its final knot."""
    stream = JointStreamer(lambda q, g: None, HZ, q0, 1-open0, VMAX, AMAX)
    stream.t_knots = np.r_[0., times]
    stream.q_knots = np.vstack([q0, qs])
    stream.g_knots = np.r_[1-open0, 1-np.asarray(openings)]
    rows = [np.r_[q0, open0]]
    for tick in range(1, int((times[-1]+10)*HZ)+1):
        t = tick/HZ
        stream.step(t)
        rows.append(np.r_[stream.qpos, 1-stream.gripper])
        if (t >= times[-1] and np.max(np.abs(stream.qpos-qs[-1])) < 1e-4
                and np.max(np.abs(stream.v)) < 1e-3
                and abs(1-stream.gripper-openings[-1]) < 1e-4):
            return np.asarray(rows)
    raise ValueError('Smooth trajectory did not settle at its final knot')


def build(manifest):
    import replay_yam as replay
    accepted = json.loads(manifest.read_text())
    if accepted.get('status') != 'human_accepted':
        raise ValueError('Expected a human-accepted manifest')
    capture = Path(accepted['capture'])
    report = accepted['review']
    if fingerprint(capture) != report['fingerprint']:
        raise ValueError('Accepted data/config/replay code changed; review again before hardware replay')
    if report.get('tracking_valid_fraction') != 1 or not report.get('planner_complete'):
        raise ValueError('Hardware replay requires complete tracking and a complete IK plan')
    profile = report['profile']
    if (profile['arm'], profile['interface']) != ('right', 'can_right'):
        raise ValueError('This entry point is for the accepted right/can_right profile')
    q0 = np.asarray(profile['start_joints_rad'], float)
    meta, images, times, grid, tracker, valid = replay.load_capture(capture, 3, .025, 30, diagnostic=True)
    if not valid.all() or not images:
        raise ValueError('Tracking or images incomplete')
    ik = PoseIK(q0)
    if not np.allclose(ik.fk(q0), profile['T_base_tool_start'], atol=1e-7):
        raise ValueError('Karma model FK no longer matches the accepted start pose')
    ext = np.asarray(profile['T_tracker_tcp'])
    targets = replay.map_targets(tracker@ext, valid, 'gravity', ik.fk(profile['anchor_joints_rad']))
    plan = Planner(ik, transit_m=.06, lift_m=.06, floor_z=.01, max_miss_mm=25).plan(q0, targets, times+.5, 0.)
    if plan['cut'] != len(plan['q']) or plan['floored']:
        raise ValueError('Full plan is not executable; refusing to truncate or change it')
    config = json.loads((ROOT/'configs/umi.review-collection.json').read_text())
    widths = replay.measure_widths(capture, images, meta, config['markers'])
    image_times = np.array([x['device_s'] for x in images])
    nearest = np.abs(grid[:, None]-image_times[None, :]).argmin(1)
    # Preserve the exact accepted preview convention: native opening = width / .095.
    # Invalid measurements hold the last visual command, initially 0.5; no invented measurements.
    opening = .5
    openings = []
    for j in nearest:
        if widths[j] is not None:
            opening = float(np.clip(widths[j]/.095, 0, 1))
        openings.append(opening)
    rows = simulate(q0, .5, plan['times'], plan['q'], np.r_[np.full(plan['transit'], .5), openings])
    if not np.isfinite(rows).all() or np.any(rows[:, :6] < ik.lower-1e-8) or np.any(rows[:, :6] > ik.upper+1e-8):
        raise ValueError('Stream violates model joint limits')
    return rows, profile, accepted


def make_rig(serial):
    from openpi_control.rigs import resolve_rig
    rig = resolve_rig('yam_bimanual').subset(['right']).with_interfaces({'right': 'can_right'})
    return replace(rig, cameras=tuple(replace(c, serial=serial) for c in rig.cameras if c.name == 'right_wrist'))


def transition(q, opening, goal, target_opening):
    duration = max(3., float(np.max(np.abs(np.asarray(goal)-q)))/.25)
    return simulate(q, opening, np.array([duration]), np.asarray(goal)[None], [target_opening])


def validate_rows(rows, bounds):
    lower, upper = bounds
    if rows.ndim != 2 or rows.shape[1] != 7 or not np.isfinite(rows).all():
        raise ValueError(f'Invalid replay array: shape={rows.shape}, expected finite (N,7)')
    bad = np.argwhere((rows[:, :6] < lower) | (rows[:, :6] > upper))
    if len(bad):
        i, j = bad[0]
        raise ValueError(f'Replay row {i}, J{j+1}={rows[i,j]:.8f} rad outside '
                         f'[{lower[j]:.8f}, {upper[j]:.8f}]')
    if np.any(rows[:, 6] < 0) or np.any(rows[:, 6] > 1):
        raise ValueError(f'Gripper commands outside [0,1]: '
                         f'{rows[:,6].min():.8f} to {rows[:,6].max():.8f}')


def startup_seed(q, opening, bounds):
    """Project only small measured stop errors; all commanded rows stay in bounds."""
    q = np.asarray(q, float)
    lower, upper = bounds
    if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(opening):
        raise ValueError('Invalid startup feedback')
    projected = np.clip(q, lower, upper)
    excess = np.abs(projected-q)
    if excess.max() > .01:
        j = int(excess.argmax())
        raise ValueError(f'Startup J{j+1}={q[j]:.8f} rad exceeds its joint limit by '
                         f'{excess[j]:.8f} rad; refusing to adjust the measured start')
    if not -.01 <= opening <= 1.01:
        raise ValueError(f'Startup gripper feedback {opening:.8f} outside allowed stop tolerance')
    if excess.any() or not 0 <= opening <= 1:
        print('Startup stop tolerance: measured joints=' + np.array2string(q, precision=8)
              + f', opening={opening:.8f}; approach commands begin at the legal boundary.', flush=True)
    return projected, float(np.clip(opening, 0, 1))


def play_hardware(rows, arm, executor, guard, limits, watch, stalled, scene, panel,
                  speed, status, phase, log, recorder=None):
    """No background command thread; parking cannot race trajectory commands."""
    from openpi_control.safety import warn_stalled_grippers
    period = 1/(HZ*speed)
    last_ui = 0.
    for index, row in enumerate(rows):
        started = time.monotonic()
        states = guard.check({'right': arm.latest_state})
        while states is None:
            time.sleep(.01)  # pause the trajectory clock; never skip rows on stale feedback
            states = guard.check({'right': arm.latest_state})
        commands = executor.step(row, limits=limits)
        arm.command(commands['right'])
        if recorder is not None:
            recorder.add(states['right'], executor.targets['right'], index)
        watch.observe(commands, states)
        warn_stalled_grippers(watch, stalled)
        now = time.monotonic()
        if now-last_ui >= 1/30 or index == len(rows)-1:
            state = states['right']
            scene.update('right', state.joints.position_rad)
            if panel is not None:
                panel.step(now-last_ui if last_ui else 1/30)
            status.content = f'**{phase}** · {index+1}/{len(rows)} · Ctrl+C parks and powers down'
            log.append({'t': now, 'phase': phase, 'index': index,
                        'command': executor.targets['right'].tolist(),
                        'measured': np.asarray(state.joints.position_rad).tolist()})
            last_ui = now
        # No catch-up bursts. A slow tick stretches time instead of skipping path points.
        time.sleep(max(0., period-(time.monotonic()-started)))


def arrival_tolerances():
    from openpi_control.config import resolve_model_assets
    config = json.loads(resolve_model_assets('Yam').model_config.read_text())
    tolerance = np.asarray([j['pos_error_margin'] for j in config['joints']], float)
    if tolerance.shape != (6,) or not np.isfinite(tolerance).all() or np.any(tolerance <= 0):
        raise ValueError('Invalid Karma joint arrival tolerances')
    return tolerance


def wait_at_goal(arm, goal, guard, timeout=5.):
    tolerance = arrival_tolerances()
    last_error = None
    deadline = time.monotonic()+timeout
    settled_since = None
    while time.monotonic() < deadline:
        states = guard.check({'right': arm.latest_state})
        last_error = None if states is None else np.abs(np.asarray(states['right'].joints.position_rad)-goal)
        if last_error is not None and np.all(last_error <= tolerance):
            settled_since = settled_since or time.monotonic()
            if time.monotonic()-settled_since > .3:
                print(f'Pose reached within Karma tolerances; max joint error {last_error.max():.5f} rad.', flush=True)
                return
        else:
            settled_since = None
        time.sleep(.01)
    detail = 'no fresh state' if last_error is None else f'joint errors {last_error.tolist()} rad; Karma tolerances {tolerance.tolist()}'
    raise RuntimeError(f'Arm did not reach the required pose: {detail}; stopping and parking')


def execute(rows, profile, rig, scene, status, speed, log, dataset_root=None, repo_id=None, provenance=None):
    from openpi_control import cameras
    from openpi_control.cli import _runtime_preflight, power_up, power_down, settle_arm_states
    from openpi_control.inference import BoundedChunkExecutor, GripperWatch, PolicyContract
    from openpi_control.safety import StateGuard, joint_limits, report_gripper_start
    from openpi_control.viz import CameraPanel
    session = None
    live = []
    readers = {}
    result = 0
    recorder = None
    completed = False
    try:
        if not _runtime_preflight(rig, camera_overrides={}):
            raise RuntimeError('Karma preflight failed; nothing energized')
        found = cameras.discover(rig.cameras)
        if found.missing:
            raise RuntimeError(f'Right wrist camera missing: {found.missing}')
        readers = cameras.open_readers(tuple(v.spec() for v in found.matched.values()))
        if any(reader.wait_for_frame(5.) is None for reader in readers.values()):
            raise RuntimeError('Right wrist camera has no frames; nothing energized')
        if dataset_root is not None:
            from yam_replay_dataset import ReplayDataset
            recorder = ReplayDataset(dataset_root, repo_id, readers, provenance=provenance)
        panel = CameraPanel(scene.server, readers, folder='Live right wrist')
        limits = joint_limits(rig)
        validate_rows(rows, limits['right'])
        session, live = power_up(rig, float_mode=False)
        arm = live[0].arm
        if not arm.capabilities.supports_move_to_ready:
            raise RuntimeError('Native home-and-shutdown capability is unavailable')
        states = settle_arm_states(live)
        report_gripper_start(states)
        if states['right'].effector is None:
            raise RuntimeError('Missing right gripper feedback')
        executor = BoundedChunkExecutor(contract=PolicyContract.from_rig(rig),
                    max_step_rad=.1*30/(HZ*speed), max_effector_step=.3*30/(HZ*speed))
        executor.reset(states)
        guard, watch, stalled = StateGuard(['right']), GripperWatch(), set()
        q = np.asarray(states['right'].joints.position_rad)
        opening = states['right'].effector.position
        log.append({'phase': 'STARTUP_FEEDBACK', 'measured': q.tolist(), 'gripper': float(opening)})
        q, opening = startup_seed(q, opening, limits['right'])
        start = transition(q, opening, profile['start_joints_rad'], .5)
        validate_rows(start, limits['right'])
        play_hardware(start, arm, executor, guard, limits, watch, stalled, scene, panel,
                      speed, status, 'APPROACH SAVED UMI START', log)
        wait_at_goal(arm, np.asarray(profile['start_joints_rad']), guard)
        play_hardware(rows, arm, executor, guard, limits, watch, stalled, scene, panel,
                      speed, status, 'REPLAY COMPLETE TRAJECTORY', log, recorder)
        wait_at_goal(arm, rows[-1, :6], guard)
        completed = True
        print(f'Completed all {len(rows)} trajectory samples; parking at native home_pos.', flush=True)
    except KeyboardInterrupt:
        print('\nCtrl+C: replay stopped; parking at native home_pos.', flush=True)
    except Exception as exc:
        result = 1
        print(f'Replay stopped: {exc}', flush=True)
    finally:
        # Same Karma native MOVE_TO_READY_AND_SHUTDOWN path as rollout.
        # First Ctrl+C gets here; a second Ctrl+C during power_down aborts parking
        # and de-energizes in place, exactly as Karma specifies.
        try:
            if session is not None:
                try:
                    status.content = '**PARKING AT NATIVE HOME → POWER DOWN**'
                finally:
                    result = max(result, power_down(session, live, park=True))
        finally:
            try:
                cameras.close_readers(readers)
            finally:
                if recorder is not None:
                    recorder.close(completed=completed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--dataset-root', type=Path, help='New LeRobot v3 directory; hardware runs default to data/lerobot/yam-replay-TIMESTAMP')
    parser.add_argument('--repo-id', default='local/yam-umi-replay')
    parser.add_argument('--execute', action='store_true', help='Enable right-arm hardware; default is offline preview')
    parser.add_argument('--speed', type=float, default=.5, help='0 < speed <= 1; default half speed')
    parser.add_argument('--camera-serial', default='254623070417')
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--check-only', action='store_true', help='Validate offline, save plan, exit')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error('--speed must be in (0, 1]')
    if args.dataset_root is not None and not args.execute:
        parser.error('--dataset-root requires --execute')
    if args.execute:
        args.dataset_root = (args.dataset_root or ROOT/'data/lerobot'/f'yam-replay-{time.time_ns()}').resolve()
        if args.dataset_root.exists():
            parser.error(f'Dataset already exists: {args.dataset_root}')
    if args.check_only and args.execute:
        parser.error('--check-only cannot be combined with --execute')
    from openpi_control.safety import joint_limits
    from openpi_control.viz import ArmSceneVisualizer
    rows, profile, accepted = build(args.manifest.resolve())
    rig = make_rig(args.camera_serial)
    validate_rows(rows, joint_limits(rig)['right'])
    out = ROOT/'artifacts/hardware-replay'/args.manifest.stem
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'trajectory.npz', commands=rows, hz=HZ, start_joints=profile['start_joints_rad'])
    summary = {'manifest': str(args.manifest.resolve()), 'arm': 'right', 'interface': 'can_right',
               'wrist_camera_serial': args.camera_serial, 'start_joints_rad': profile['start_joints_rad'],
               'samples': len(rows), 'replay_seconds': (len(rows)-1)/HZ/args.speed,
               'speed': args.speed, 'missing_width_policy': 'hold previous accepted preview command',
               'gripper_mapping': 'native opening = width_m / 0.095; nominal, not physically calibrated',
               'manual_override': accepted.get('manual_override'), 'park': 'Karma native home_pos then shutdown'}
    (out/'plan.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    if args.check_only:
        return 0
    scene = ArmSceneVisualizer.from_rig(rig, port=args.port)
    status = scene.server.gui.add_markdown('**OFFLINE PREVIEW — no hardware connected**')
    scene.update('right', profile['start_joints_rad'])
    # Show the complete path once, at the same density as the accepted 30-Hz view.
    scene.update_chunk({'right': rows[::7, :6]})
    scene.server.gui.add_markdown('Saved UMI start → complete accepted trajectory → native home on hardware.\n\n'
        'Gripper holds its previous command where jaw measurements are missing. '
        'The preview does not model the native parking motion or collisions.')
    log = []
    try:
        if args.execute:
            return execute(rows, profile, rig, scene, status, args.speed, log,
                           dataset_root=args.dataset_root, repo_id=args.repo_id, provenance=summary)
        cursor = 0.
        last = time.monotonic()
        while True:
            now = time.monotonic()
            cursor = (cursor+(now-last)*HZ*args.speed) % len(rows)
            last = now
            scene.update('right', rows[int(cursor), :6])
            status.content = f'**OFFLINE PREVIEW** · {cursor/HZ:.1f}/{(len(rows)-1)/HZ:.1f} s · speed {args.speed:g}'
            time.sleep(1/30)
    except KeyboardInterrupt:
        return 0
    finally:
        scene.server.stop()
        if args.execute:
            (out/f'execution-{time.time_ns()}.json').write_text(json.dumps(log)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
