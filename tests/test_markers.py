import cv2
import numpy as np
import pytest

from yumi.markers import Markers


def test_metric_markers_and_occlusion():
    # Render two 20 mm tags, 60 mm center spacing at 300 mm range.
    config = {
        "dictionary": "DICT_4X4_50",
        "ids": [13, 14],
        "size_m": 0.02,
        "min_side_px": 16,
        "max_reprojection_px": 1,
        "max_ambiguity_m": 0.002,
        "width_gain": 1.0,
        "width_offset_m": -0.02,
        "width_range_m": [0, 0.1],
    }
    intr = {
        "fx": 600.0,
        "fy": 600.0,
        "ppx": 320.0,
        "ppy": 240.0,
        "coeffs": [0.0] * 5,
        "model": "distortion.none",
    }
    rgb = np.full((480, 640, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker, x in [(13, 240), (14, 360)]:
        tag = cv2.aruco.generateImageMarker(dictionary, marker, 40)
        rgb[220:260, x : x + 40] = tag[:, :, None]
    d = Markers(config, intr)
    width, error, _ = d.width(rgb)
    assert width == pytest.approx(0.04, abs=0.002)
    assert error < 0.2
    rgb[220:260, 240:280] = 255
    with pytest.raises(ValueError, match="Missing"):
        d.width(rgb)


def test_image_width_calibration_and_missing_marker_rejection():
    from yumi.calibration import fit_image_width

    x = np.linspace(80, 280, 6)
    fit = fit_image_width(
        {
            "marker_separation_px": x.tolist(),
            "jaw_opening_m": ((x - 80) * 0.0002).tolist(),
            "calibration_identity": {"resolution": [640, 480]},
        }
    )
    config = {
        "dictionary": "DICT_4X4_50",
        "ids": [13, 14],
        "size_m": 0.01,
        "min_side_px": 16,
        "width_method": "image_calibrated",
        "image_width_calibration": fit["image_width_calibration"],
        "width_range_m": [0, 0.1],
    }
    intr = {
        "fx": 600.0,
        "fy": 600.0,
        "ppx": 320.0,
        "ppy": 240.0,
        "coeffs": [0.0] * 5,
        "model": "distortion.none",
    }
    rgb = np.full((480, 640, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker, left in [(13, 200), (14, 380)]:
        rgb[200:240, left : left + 40] = cv2.aruco.generateImageMarker(
            dictionary, marker, 40
        )[:, :, None]
    d = Markers(config, intr)
    width, error, _ = d.width(rgb)
    assert width == pytest.approx(0.020, abs=1e-5)
    assert error == -1  # No invented 3D reprojection score for a 2D measurement.
    with pytest.raises(ValueError, match="Resolution"):
        d.width(np.pad(rgb, ((0, 10), (0, 0), (0, 0)), constant_values=255))
    rgb[200:240, 200:240] = 255
    with pytest.raises(ValueError, match="Missing"):
        d.width(rgb)


def test_image_width_fit_rejects_bad_physical_measurements():
    from yumi.calibration import fit_image_width

    x = np.linspace(80, 280, 6)
    y = (x - 80) * 0.0002
    y[2] += 0.01
    with pytest.raises(ValueError, match="exceeds"):
        fit_image_width(
            {
                "marker_separation_px": x.tolist(),
                "jaw_opening_m": y.tolist(),
                "calibration_identity": {"resolution": [640, 480]},
            }
        )


def test_projective_center_not_corner_average():
    d = Markers(
        {
            "dictionary": "DICT_4X4_50",
            "ids": [13, 14],
            "size_m": 0.01,
            "min_side_px": 1,
        },
        {
            "fx": 600.0,
            "fy": 600.0,
            "ppx": 320.0,
            "ppy": 240.0,
            "coeffs": [0.0] * 5,
            "model": "distortion.none",
        },
    )
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    h = np.array([[40, 0, 100], [0, 60, 100], [0.3, 0.1, 1.0]])
    pts = cv2.perspectiveTransform(square[None], h)

    class Detector:
        def detectMarkers(self, _image):
            return [pts, pts + np.array([[[150.0, 0.0]]])], np.array([[13], [14]]), []

    d.detector = Detector()
    separation, _, centers = d.image_separation(np.zeros((480, 640, 3), np.uint8))
    expected = cv2.perspectiveTransform(np.array([[[0.5, 0.5]]], np.float32), h)[0, 0]
    np.testing.assert_allclose(centers[0], expected, atol=1e-5)
    assert separation == pytest.approx(150.0)


def test_provisional_fit_keeps_error_and_does_not_change_default_gate():
    from yumi.calibration import fit_image_width

    x = np.linspace(80, 280, 6)
    y = (x - 80) * 0.0002
    y[2] += 0.004
    samples = {
        "marker_separation_px": x.tolist(),
        "jaw_opening_m": y.tolist(),
        "calibration_identity": {"resolution": [640, 480]},
    }
    with pytest.raises(ValueError, match="exceeds"):
        fit_image_width(samples)
    result = fit_image_width(samples, max_error_m=0.0035)["image_width_calibration"]
    assert result["validation_status"] == "provisional_pipeline_test"
    assert 0.001 < result["max_training_error_m"] <= 0.0035
