"""Offline workflow tests; no cameras, CAN interfaces or robot connections."""
import importlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
workflow = importlib.import_module('collect_umi_run')
controls = importlib.import_module('umi_review_controls')


@pytest.mark.parametrize('actions', [('retry', 'accept_next', 'accept_finish'), ('finish',)])
def test_record_review_decision_loop(tmp_path, monkeypatch, actions):
    calls = []
    monkeypatch.setattr(workflow, 'require_free_port', lambda port: None)

    def capture(run, port):
        result = run/f'episode-{len(calls):03d}'
        calls.append(('record', result))
        return result

    choices = iter(actions)

    def review(episode, run, port):
        calls.append(('review', episode))
        return next(choices)

    monkeypatch.setattr(workflow, 'capture_once', capture)
    monkeypatch.setattr(workflow, 'review_once', review)
    workflow.collect(tmp_path, 8080)
    assert [kind for kind, _ in calls] == ['record', 'review']*len(actions)
    assert all(calls[i][1] == calls[i+1][1] for i in range(0, len(calls), 2))


def test_quit_idle_does_not_review(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'require_free_port', lambda port: None)
    monkeypatch.setattr(workflow, 'capture_once', lambda *args: None)
    monkeypatch.setattr(workflow, 'review_once', lambda *args: pytest.fail('No capture to review'))
    workflow.collect(tmp_path, 8080)


def test_failed_review_keeps_attempt_and_does_not_accept(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'require_free_port', lambda port: None)
    episode = tmp_path/'episode-000'
    episode.mkdir()
    (episode/'COMPLETE').touch()

    def fail(*args):
        raise RuntimeError('Tracking failed')

    monkeypatch.setattr(workflow, 'review_once', fail)
    monkeypatch.setattr('builtins.input', lambda _: 'q')
    workflow.collect(tmp_path, 8080, episode)
    assert (episode/'COMPLETE').exists()
    assert not (tmp_path/'accepted').exists()


class Handle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def on_click(self, callback):
        self.click = lambda: callback(None)


class FakeGUI:
    def __init__(self):
        self.buttons = {}

    def add_folder(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_markdown(self, content):
        self.status = Handle(content=content)
        return self.status

    def add_button(self, label, **kwargs):
        button = Handle(disabled=kwargs.get('disabled', False))
        self.buttons[label] = button
        return button


def setup_controls(tmp_path, passed=True):
    gui = FakeGUI()
    stop = threading.Event()
    decision = tmp_path/'decision.json'
    report = {'review_gate_passed': passed, 'recorded_tracked_fraction': .98, 'width_valid_fraction': .99}
    controls.add_review_controls(SimpleNamespace(gui=gui), tmp_path/'capture', decision, report, stop)
    return gui, stop, decision


def test_accept_writes_manifest_before_next_and_only_once(tmp_path, monkeypatch):
    accepted = []
    monkeypatch.setattr(controls, 'accept', lambda *args: accepted.append(args) or tmp_path/'accepted.json')
    gui, stop, decision = setup_controls(tmp_path)
    gui.buttons['Accept & next'].click()
    gui.buttons['Accept & next'].click()
    assert len(accepted) == 1
    assert stop.is_set()
    assert json.loads(decision.read_text())['action'] == 'accept_next'
    assert all(b.disabled for b in gui.buttons.values())


def test_accept_failure_keeps_review_open(tmp_path, monkeypatch):
    def stale(*args):
        raise ValueError('Changed capture; review again')
    monkeypatch.setattr(controls, 'accept', stale)
    gui, stop, decision = setup_controls(tmp_path)
    gui.buttons['Accept & next'].click()
    assert not stop.is_set()
    assert not decision.exists()
    assert 'Changed capture' in gui.status.content


@pytest.mark.parametrize('label,action', [
    ('Retry (keep this attempt pending)', 'retry'),
    ('Finish (keep this attempt pending)', 'finish'),
])
def test_failed_gate_allows_retry_or_finish_without_acceptance(tmp_path, monkeypatch, label, action):
    monkeypatch.setattr(controls, 'accept', lambda *args: pytest.fail('Must not accept'))
    gui, stop, decision = setup_controls(tmp_path, passed=False)
    assert gui.buttons['Accept & next'].disabled
    assert gui.buttons['Accept & finish'].disabled
    gui.buttons[label].click()
    assert stop.is_set()
    assert json.loads(decision.read_text())['action'] == action


def test_max_episodes_rejects_noninteractive_before_opening_sensors(tmp_path, monkeypatch):
    from yumi import hardware
    monkeypatch.setattr(hardware, 'sdk', lambda: pytest.fail('Must not open sensors'))
    with pytest.raises(ValueError, match='max_episodes'):
        hardware.record({}, tmp_path, max_episodes=1)


def test_ctrl_c_does_not_send_second_interrupt_to_child(monkeypatch):
    calls = []

    class Child:
        def wait(self, timeout=None):
            calls.append(('wait', timeout))
            if timeout is None:
                raise KeyboardInterrupt
            return 0

        def poll(self):
            return None

        def send_signal(self, sig):
            pytest.fail('Terminal already delivered SIGINT to the child')

    monkeypatch.setattr(workflow.subprocess, 'Popen', lambda *args, **kwargs: Child())
    with pytest.raises(KeyboardInterrupt):
        workflow.run_child(['unused'])
    assert calls == [('wait', None), ('wait', 15)]


def test_sensor_cleanup_ignores_repeat_interrupt_and_restores_handler():
    import os
    import signal
    from yumi.hardware import _uninterrupted_cleanup

    previous = signal.getsignal(signal.SIGINT)
    with _uninterrupted_cleanup():
        os.kill(os.getpid(), signal.SIGINT)
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == previous
