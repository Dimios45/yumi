import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('umi_review_state', Path(__file__).resolve().parents[1]/'scripts/umi_review_state.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_accept_requires_checks_and_unchanged_capture(tmp_path):
    capture = tmp_path/'run'/'episode-000'
    capture.mkdir(parents=True)
    (capture/'COMPLETE').touch()
    (capture/'samples.jsonl').write_text('raw data')
    report = {'review_gate_passed': False, 'fingerprint': m.fingerprint(capture)}
    path = capture/'umi-review.json'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='checks failed'):
        m.accept(capture, tmp_path/'accepted')
    report['review_gate_passed'] = True
    path.write_text(json.dumps(report))
    (capture/'samples.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        m.accept(capture, tmp_path/'accepted')
    report['fingerprint'] = m.fingerprint(capture)
    path.write_text(json.dumps(report))
    result = m.accept(capture, tmp_path/'accepted')
    assert json.loads(result.read_text())['status'] == 'human_accepted'
    assert (capture/'samples.jsonl').read_text() == 'changed'
    with pytest.raises(FileExistsError):
        m.accept(capture, tmp_path/'accepted')


def test_explicit_override_keeps_failed_checks_and_still_checks_fingerprint(tmp_path):
    capture = tmp_path/'run'/'episode-001'
    capture.mkdir(parents=True)
    (capture/'samples.jsonl').write_text('original')
    report = {'review_gate_passed': False, 'width_valid_fraction': .642,
              'position_error_mm': {'max': 53.1}, 'fingerprint': m.fingerprint(capture)}
    (capture/'umi-review.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='nonempty'):
        m.accept(capture, tmp_path/'accepted', override_reason=' ')
    result = m.accept(capture, tmp_path/'accepted', override_reason='User approved after visual review')
    payload = json.loads(result.read_text())
    assert payload['review'] == report
    assert payload['manual_override']['automatic_checks_passed'] is False
    assert payload['manual_override']['missing_measurements_remain_invalid'] is True
    (capture/'samples.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        m.accept(capture, tmp_path/'elsewhere', override_reason='User approved')
