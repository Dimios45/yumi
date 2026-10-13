"""Karma environment only. All hardware functions replaced before execution."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('openpi_control')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import replay_accepted_yam as r


@pytest.mark.parametrize('failure', [None, KeyboardInterrupt, RuntimeError])
@pytest.mark.parametrize('record_dataset', [False, True])
def test_completion_interrupt_and_error_park_right_arm(monkeypatch, failure, record_dataset):
    from openpi_control import cli, cameras, viz
    calls = []
    state = NS(joints=NS(position_rad=np.array([0., .7, .7, 0, 0, 0])), effector=NS(position=.5))
    arm = NS(capabilities=NS(supports_move_to_ready=True))
    session = object()
    monkeypatch.setattr(cli, '_runtime_preflight', lambda *a, **kw: True)
    monkeypatch.setattr(cameras, 'discover', lambda *a: NS(missing={}, matched={}))
    monkeypatch.setattr(cameras, 'open_readers', lambda *a: {})
    monkeypatch.setattr(cameras, 'close_readers', lambda *a: calls.append('camera_close'))
    monkeypatch.setattr(viz, 'CameraPanel', lambda *a, **kw: None)
    monkeypatch.setattr(cli, 'power_up', lambda *a, **kw: (session, [NS(arm=arm, name='right')]))
    monkeypatch.setattr(cli, 'settle_arm_states', lambda *a: {'right': state})
    def park(s, arms, *, park):
        assert s is session and len(arms) == 1 and arms[0].name == 'right' and park
        calls.append('park')
        return 0
    monkeypatch.setattr(cli, 'power_down', park)
    q = state.joints.position_rad
    rows = np.array([np.r_[q, .5], np.r_[q, .6]])
    monkeypatch.setattr(r, 'transition', lambda *a: rows[:1])
    monkeypatch.setattr(r, 'wait_at_goal', lambda *a: None)
    def play(*args):
        calls.append(('play', len(args[0])))
        if failure is not None:
            raise failure('test')
    monkeypatch.setattr(r, 'play_hardware', play)
    import yam_replay_dataset
    recorder = NS(close=lambda completed=False: calls.append(('dataset_close', completed)))
    monkeypatch.setattr(yam_replay_dataset, 'ReplayDataset', lambda *a, **kw: recorder)
    result = r.execute(rows, {'start_joints_rad': q}, r.make_rig('352122271723'), NS(server=None), NS(content=''), .5, [],
                       dataset_root=Path('/unused-test-dataset') if record_dataset else None, repo_id='local/test')
    if record_dataset:
        assert calls[-3:] == ['park', 'camera_close', ('dataset_close', failure is None)]
    else:
        assert calls[-2:] == ['park', 'camera_close']
    assert calls.count('park') == 1
    if failure is None:
        assert calls[:2] == [('play', 1), ('play', 2)]
    assert result == (1 if failure is RuntimeError else 0)


def test_stream_reaches_last_sample_and_bounds():
    q0 = np.array([0., .7, .7, 0, 0, 0])
    goal = q0+np.array([.05, -.03, .02, .1, .05, -.1])
    rows = r.simulate(q0, .5, np.array([1.]), goal[None], [.6])
    assert np.max(np.abs(rows[-1, :6]-goal)) < 1e-4
    assert abs(rows[-1, 6]-.6) < 1e-4
    assert np.all(np.abs(np.diff(rows[:, :6],axis=0))*r.HZ <= r.VMAX+1e-8)
    assert r.make_rig('352122271723').names == ('right',)
    assert r.make_rig('352122271723').camera_names == ('right_wrist',)


def test_near_zero_encoder_error_does_not_make_invalid_commands():
    from openpi_control.safety import joint_limits
    bounds = joint_limits(r.make_rig('254623070417'))['right']
    measured = np.array([0., -.001, -.002, 0., 0., 0.])
    q, opening = r.startup_seed(measured, 1.0001, bounds)
    assert q[1] == 0 and q[2] == 0 and opening == 1
    goal = np.array([0., .681154, .564064, .250521, 0., .027388])
    approach = r.transition(q, opening, goal, .5)
    r.validate_rows(approach, bounds)
    assert np.max(abs(approach[-1,:6]-goal)) < 1e-4
    with pytest.raises(ValueError, match='Startup J2'):
        r.startup_seed(np.array([0., -.02, 0., 0., 0., 0.]), 1., bounds)
    with pytest.raises(ValueError, match='J2='):
        r.validate_rows(np.array([np.r_[measured, 1.]]), bounds)


@pytest.mark.parametrize('error,passes', [(.041253, True), (.060, False)])
def test_arrival_uses_karma_tolerance(monkeypatch, error, passes):
    q = np.array([0., .681154, .564064, .250521, 0., .027388])
    measured = q.copy()
    measured[2] -= error
    state = NS(joints=NS(position_rad=measured))
    arm = NS(latest_state=state)
    guard = NS(check=lambda _: {'right': state})
    clock = [0.]
    monkeypatch.setattr(r.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(r.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    assert np.allclose(r.arrival_tolerances(), .05)
    if passes:
        r.wait_at_goal(arm, q, guard, timeout=.5)
    else:
        with pytest.raises(RuntimeError, match='joint errors'):
            r.wait_at_goal(arm, q, guard, timeout=.5)
