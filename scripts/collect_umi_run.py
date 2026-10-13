#!/usr/bin/env python3
"""Record → Viser review → accept/retry, with one command and no robot connection."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PYTHON = ROOT/'.venv/bin/python'
REPLAY_PYTHON = ROOT.parent/'karma/.venv/bin/python'


def run_child(command):
    # Inherit the terminal for the existing recorder's SPACE/q controls.
    process = subprocess.Popen([str(x) for x in command], cwd=ROOT)
    interrupted = False
    try:
        return process.wait()
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if process.poll() is None:
            # Terminal Ctrl+C already reaches the child in our process group.
            # A second SIGINT would interrupt camera/writer cleanup.
            if not interrupted:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def require_free_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
        except OSError as exc:
            raise RuntimeError(f'Port {port} is in use. Stop its viewer or use --port with a free port.') from exc


def capture_once(run, port):
    before = set(run.glob('episode-*'))
    code = run_child([CAPTURE_PYTHON, Path(__file__).resolve(), '--capture-worker', run, '--port', port])
    if code:
        raise RuntimeError(f'Capture exited with status {code}; raw attempt retained at {run}')
    captures = sorted(p for p in set(run.glob('episode-*'))-before if (p/'COMPLETE').exists())
    if len(captures) > 1:
        raise RuntimeError('Expected one capture; retained episodes require individual review')
    return captures[0] if captures else None


def review_once(capture, run, port):
    decision = run/f'review-decision-{uuid.uuid4().hex}.json'
    code = run_child([REPLAY_PYTHON, ROOT/'scripts/review_umi_run.py', capture,
                      '--port', port, '--decision-file', decision])
    if code:
        raise RuntimeError(f'Review exited with status {code}; {capture} remains on disk')
    if not decision.exists():
        return 'finish'
    result = json.loads(decision.read_text())
    action = result['action']
    if action not in ('accept_next', 'accept_finish', 'retry', 'finish'):
        raise ValueError(f'Unknown review decision: {action}')
    if result.get('manifest'):
        print(f"Accepted: {result['manifest']}", flush=True)
    return action


def collect(run, port, first_capture=None):
    capture = first_capture
    while True:
        try:
            require_free_port(port)
            if capture is None:
                print(f'\nRECORD · http://192.168.0.196:{port}\n'
                      'Refresh Viser. Warm up to confidence 3. SPACE starts; SPACE stops and opens review. q exits.', flush=True)
                capture = capture_once(run, port)
                if capture is None:
                    return
            print(f'\nREVIEW · {capture}\nComputing the smooth replay. '
                  f'Refresh http://192.168.0.196:{port} when the viewer is ready.\n'
                  'Use Accept & next, Retry, or Finish in Viser.', flush=True)
            action = review_once(capture, run, port)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f'\n{exc}', flush=True)
            answer = input('Attempt retained. Enter r to record another attempt, or q to finish: ').strip().lower()
            if answer != 'r':
                return
            action = 'retry'
        if action in ('finish', 'accept_finish'):
            return
        capture = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080, help='Same Viser port for capture and replay')
    parser.add_argument('--review', type=Path, help='Resume by reviewing an existing raw episode')
    parser.add_argument('--capture-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture_worker is not None:
        sys.path.insert(0, str(ROOT/'src'))
        from yumi.hardware import record
        config = json.loads((ROOT/'configs/umi.review-collection.json').read_text())
        record(config, args.capture_worker, interactive=True, port=args.port,
               world_serial=config.get('world_serial'), max_episodes=1,
               require_tracker_confidence=3)
        return
    for python in (CAPTURE_PYTHON, REPLAY_PYTHON):
        if not python.is_file():
            parser.error(f'Missing environment: {python}')
    capture = args.review.resolve() if args.review else None
    if capture and not (capture/'COMPLETE').exists():
        parser.error('--review requires a complete raw episode')
    run = ROOT/'data/umi-runs/pending'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    print(f'UMI collection · raw attempts: {run}\nOnly explicit acceptance adds an accepted manifest.', flush=True)
    try:
        collect(run, args.port, capture)
    except (KeyboardInterrupt, EOFError):
        print('\nStopped. Unaccepted attempts remain pending.', flush=True)


if __name__ == '__main__':
    main()
