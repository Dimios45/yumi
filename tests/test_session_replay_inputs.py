import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'session_replay_inputs',
    Path(__file__).resolve().parents[1] / 'scripts/session_replay_inputs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
replay_widths = module.replay_widths


def rows(widths, times=None):
    if times is None:
        times = np.arange(len(widths)) / 30
    return [{'device_s': t, 'width_measurement': {
        'valid': w is not None, 'opening_m': w}}
        for t, w in zip(times, widths)]


def test_adjacent_measurements_and_exact_endpoints():
    result = replay_widths(rows([0, .03, .05]), [0, 1/60, 2/30], [0, .053], .05)
    assert result == pytest.approx([0, .015, .05])


@pytest.mark.parametrize('widths', [[.01, None, .03], [None, None, None], [.01, .06, .03]])
def test_missing_or_out_of_range_widths_are_not_filled(widths):
    with pytest.raises(ValueError, match='lack adjacent valid'):
        replay_widths(rows(widths), [1/30], [0, .053], .05)


@pytest.mark.parametrize('query', [[-.01], [.08]])
def test_no_endpoint_extrapolation(query):
    with pytest.raises(ValueError, match='lack adjacent valid'):
        replay_widths(rows([0, .02, .04]), query, [0, .053], .05)


def test_dropped_images_and_timestamp_reset():
    with pytest.raises(ValueError, match='lack adjacent valid'):
        replay_widths(rows([.01, .02], [0, .1]), [.05], [0, .053], .05)
    with pytest.raises(ValueError, match='strictly increasing'):
        replay_widths(rows([.01, .02], [0, 0]), [0], [0, .053], .05)
