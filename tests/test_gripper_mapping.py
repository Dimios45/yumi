import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'gripper_mapping', Path(__file__).resolve().parents[1] / 'scripts/gripper_mapping.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def calibration():
    return {'arm': 'right', 'interface': 'can_right', 'points': [
        {'gap_mm': 0, 'command': 0, 'measured': .01},
        {'gap_mm': 40, 'command': .5, 'measured': .48},
        {'gap_mm': 95, 'command': 1, 'measured': .98}]}


def test_metric_aperture_preserved_with_wider_robot_and_separate_feedback():
    mapping = module.MeasuredGripperMapping(calibration(), 'right', 'can_right')
    assert mapping.width_to_open(.04) == .5
    assert mapping.width_to_open(.053) == pytest.approx(.5+13/55*.5)
    assert mapping.feedback_to_width(.48) == .04
    assert mapping.command_to_width(.5) == .04
    widths = np.linspace(0, .095, 31)
    assert mapping.command_to_width(mapping.width_to_open(widths)) == pytest.approx(widths)


@pytest.mark.parametrize('value', [-.001, .096, float('nan')])
def test_no_extrapolation_or_nonfinite_commands(value):
    mapping = module.MeasuredGripperMapping(calibration(), 'right', 'can_right')
    with pytest.raises(ValueError, match='outside measured'):
        mapping.width_to_open(value)


def test_reject_wrong_arm_and_nonmonotonic_measurements():
    doc = calibration()
    with pytest.raises(ValueError, match='does not match'):
        module.MeasuredGripperMapping(doc, 'left', 'can_left')
    doc['points'][1]['measured'] = .99
    with pytest.raises(ValueError, match='strictly increasing'):
        module.MeasuredGripperMapping(doc, 'right', 'can_right')


def test_reject_feedback_outside_measured_range():
    mapping = module.MeasuredGripperMapping(calibration(), 'right', 'can_right')
    with pytest.raises(ValueError, match='feedback outside'):
        mapping.feedback_to_width(1)
