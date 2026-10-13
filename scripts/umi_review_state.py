"""Content-bound acceptance of raw UMI captures. Does not operate cameras or robots."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(capture):
    h = hashlib.sha256()
    files = [p for p in capture.rglob('*') if p.is_file() and p.name != 'umi-review.json']
    files += [ROOT/'configs/yam.umi-start.json', ROOT/'configs/umi.review-collection.json',
              ROOT/'scripts/review_umi_run.py', ROOT/'scripts/yam_motion.py', ROOT/'scripts/replay_yam.py']
    for p in sorted(files):
        h.update(str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).encode())
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
    return h.hexdigest()


def accept(capture, destination, *, override_reason=None):
    report = json.loads((capture/'umi-review.json').read_text())
    if override_reason is not None and not override_reason.strip():
        raise ValueError('Manual acceptance requires a nonempty reason.')
    if not report.get('review_gate_passed') and override_reason is None:
        raise ValueError('Trajectory/width checks failed; retain as pending and inspect umi-review.json.')
    if report['fingerprint'] != fingerprint(capture):
        raise ValueError('Capture, profile or replay code changed. Review again before acceptance.')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination/(capture.parent.name+'--'+capture.name+'.json')
    payload = {'status': 'human_accepted', 'capture': str(capture),
               'accepted_at': datetime.now(timezone.utc).isoformat(), 'review': report,
               'physical_calibration_verified': False,
               'note': 'Accepted offline trajectory; physical calibration is still provisional.'}
    if override_reason is not None:
        payload['manual_override'] = {
            'reason': override_reason,
            'automatic_checks_passed': bool(report.get('review_gate_passed')),
            'missing_measurements_remain_invalid': True,
        }
    with manifest.open('x') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--accept', action='store_true', required=True,
                        help='Confirm you inspected the replay and want this demonstration accepted')
    args = parser.parse_args()
    try:
        print(accept(args.capture.resolve(), ROOT/'data/umi-runs/accepted'))
    except (ValueError, FileExistsError) as exc:
        parser.exit(1, str(exc)+'\n')
