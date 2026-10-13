"""Run in Karma's environment, where the offline replay dependencies exist."""
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip('mujoco')
pytest.importorskip('vr_teleop_kit')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_yam import load_capture


def capture(tmp_path, confidence):
    (tmp_path/'metadata.json').write_text('{}')
    rows = [dict(kind='pose', domain='hardware_clock', device_s=i*.01,
                 tracker_confidence=c, position=[i*.001, 0, 0],
                 quaternion_xyzw=[0, 0, 0, 1]) for i, c in enumerate(confidence)]
    (tmp_path/'samples.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    return tmp_path


def test_low_confidence_is_diagnostic_and_never_valid(tmp_path):
    root = capture(tmp_path, [2]*100)
    with pytest.raises(ValueError, match='No tracker sample'):
        load_capture(root, 3, .025, 30)
    result = load_capture(root, 3, .025, 30, diagnostic=True)
    assert len(result[2]) > 20
    assert not result[-1].any()


def test_diagnostic_review_does_not_discard_low_confidence_prefix(tmp_path):
    root = capture(tmp_path, [2]*50+[3]*50)
    result = load_capture(root, 3, .025, 30, diagnostic=True)
    assert result[3][0] == 0
    assert not result[-1][0]
    assert result[-1][-1]
