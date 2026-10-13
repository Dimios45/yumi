import numpy as np
import pytest

from yumi.calibration import fit_piecewise_image_width
from yumi.markers import Markers


def samples():
    return {
        "marker_separation_px": [62, 170, 280, 400],
        "jaw_opening_m": [0, 0.018, 0.036, 0.055],
        "calibration_identity": {"resolution": [640, 480]},
    }


def test_four_point_calibration_requires_explicit_opt_in_and_preserves_scale():
    with pytest.raises(ValueError, match="at least 6"):
        fit_piecewise_image_width(samples())
    fit = fit_piecewise_image_width(samples(), minimum_samples=4)
    detector = object.__new__(Markers)
    detector.c = {**fit, "width_range_m": [0, 0.055]}
    assert detector.opening_from_separation(62, [640, 480]) == 0
    assert detector.opening_from_separation(400, [640, 480]) == 0.055
    assert detector.opening_from_separation(225, [640, 480]) == pytest.approx(0.027)
    assert (
        fit["image_width_calibration"]["validation_status"]
        == "measured_lookup_not_independently_validated"
    )
    with pytest.raises(ValueError, match="outside measured"):
        detector.opening_from_separation(410, [640, 480])


def test_mislabeled_or_insufficient_settings_are_rejected():
    doc = samples()
    doc["jaw_opening_m"] = [0, 0.036, 0.018, 0.055]
    with pytest.raises(ValueError, match="increase together"):
        fit_piecewise_image_width(doc, minimum_samples=4)
    doc["jaw_opening_m"] = [0, 0.018, np.nan, 0.055]
    with pytest.raises(ValueError, match="finite paired"):
        fit_piecewise_image_width(doc, minimum_samples=4)
