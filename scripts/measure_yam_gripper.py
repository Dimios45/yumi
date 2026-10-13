#!/usr/bin/env python3
"""Measure the YAM jaw gap (mm) against its normalized gripper command (1 = open).

The arm is energized and held where it powered up; only the gripper moves, slowly.
At each command, measure the gap between the inner faces of the fingertips with
calipers and type it in mm (Enter alone skips a point). Then the touch step closes
the jaws in small steps until they just touch the block.

The result goes to configs/yam.gripper-width.json, and a straight-line fit is
printed as the --umi-width-range for infer_act_relative.py.

Run from ~/karma:
  uv run python ../yumi/scripts/measure_yam_gripper.py
"""

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', default='right')
    p.add_argument('--interface', default='can_right')
    p.add_argument('--commands', type=float, nargs='+', default=[1., .875, .75, .625, .5, .375, .25, .125, 0.])
    p.add_argument('--touch-step', type=float, default=.02, help='command step while closing onto the block')
    p.add_argument('--rate', type=float, default=.3, help='gripper ramp speed, command units per second')
    p.add_argument('--output', type=Path, default=ROOT/'configs/yam.gripper-width.json')
    p.add_argument('--no-park', dest='park', action='store_false', help='power down in place')
    args = p.parse_args()

    from openpi_control.cli import power_down, power_up, settle_arm_states
    from openpi_control.rigs import resolve_rig
    from openpi_control.safety import MAX_STATE_AGE_S
    from openpi_control.types import PositionCommand

    rig = resolve_rig('yam_bimanual').subset([args.arm]).with_interfaces({args.arm: args.interface})
    session, live_arms = power_up(rig)
    arm = live_arms[0].arm
    stop, fault, points = threading.Event(), [], []
    try:
        state = settle_arm_states(live_arms)[args.arm]
        q0 = np.asarray(state.joints.position_rad, float)[:6]  # held for the whole run
        target = {'e': state.effector.position if state.effector is not None else 1.}
        sent = {'e': target['e']}

        def stream():  # 50 Hz: same joints, gripper ramped toward the target
            try:
                while not stop.wait(.02):
                    s = arm.latest_state
                    if s is None or not s.is_fresh(MAX_STATE_AGE_S):
                        raise RuntimeError('arm state went stale')
                    sent['e'] += float(np.clip(target['e']-sent['e'], -args.rate*.02, args.rate*.02))
                    arm.command(PositionCommand(q0, effector=sent['e']))
            except Exception as err:  # noqa: BLE001 - reported by the main loop, which parks
                fault.append(err)
                stop.set()

        def measured():
            s = arm.latest_state
            return None if s is None or s.effector is None else float(s.effector.position)

        def go(e):
            """Ramp to command e, then wait until the measured opening stops changing."""
            target['e'] = float(np.clip(e, 0, 1))
            while abs(sent['e']-target['e']) > 1e-4 and not stop.is_set():
                time.sleep(.02)
            last, still = measured(), 0
            for _ in range(150):  # up to 3 s to settle
                time.sleep(.02)
                m = measured()
                still = still+1 if m is not None and last is not None and abs(m-last) < 5e-4 else 0
                last = m
                if still >= 15:
                    break
            if fault:
                raise RuntimeError(f'gripper stream: {fault[0]}')
            return measured()

        def ask(prompt):
            answer = input(prompt).strip().lower()
            if answer == 'q':
                raise KeyboardInterrupt
            return answer

        threading.Thread(target=stream, daemon=True).start()
        print(f'arm held at {np.round(q0, 3).tolist()} rad; gripper moves at {args.rate} per second. q quits.')
        for e in args.commands:
            m = go(e)
            answer = ask(f'command {e:.3f} (measured {m:.3f}): gap in mm, Enter to skip: ')
            if answer:
                points.append({'command': e, 'measured': m, 'gap_mm': float(answer)})

        touch = None
        go(1.)
        ask('Touch: hold the block between the fingertips, then press Enter. ')
        e = 1.
        while e > 0:
            e = round(e-args.touch_step, 4)
            m = go(e)
            if ask(f'command {e:.3f} (measured {m:.3f}): Enter = close more, t = touching: ') == 't':
                touch = {'command': e, 'measured': m}
                break
        go(1.)

        result = {'arm': args.arm, 'interface': args.interface, 'date': time.strftime('%Y-%m-%d %H:%M'),
                  'points': points, 'block_touch': touch}
        if len(points) >= 2:
            c = np.array([pt['command'] for pt in points])
            g = np.array([pt['gap_mm'] for pt in points])
            slope, closed = np.polyfit(c, g, 1)
            rms = float(np.sqrt(np.mean((closed+slope*c-g)**2)))
            result['linear_fit'] = {'gap_mm_at_closed': float(closed), 'gap_mm_at_open': float(closed+slope),
                                    'rms_mm': rms}
            print(f'\nfit: gap = {closed:.1f} mm + {slope:.1f} mm x command  (rms {rms:.2f} mm)')
            print(f'  --umi-width-range {closed/1000:.4f} {(closed+slope)/1000:.4f}   (gaps match the UMI exactly)')
            if rms > 1.:
                print('  rms > 1 mm: the jaw is not linear in the command; the table above should be '
                      'interpolated instead of using one range')
            if touch is not None:
                print(f'  block touches at command {touch["command"]:.3f} '
                      f'(fit says {closed+slope*touch["command"]:.1f} mm)')
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(f'saved {args.output}')
    except KeyboardInterrupt:
        print('\nstopped' + (f'; readings so far: {json.dumps(points)}' if points else ''))
    finally:
        stop.set()
        power_down(session, live_arms, park=args.park)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
