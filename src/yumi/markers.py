"""Metric two-marker aperture. Tiny planar markers are intrinsically noisy."""

import cv2
import numpy as np


def camera_matrix(intrinsics):
    return np.array(
        [
            [intrinsics["fx"], 0, intrinsics["ppx"]],
            [0, intrinsics["fy"], intrinsics["ppy"]],
            [0, 0, 1.0],
        ],
        float,
    )


def rectify_points(points, intrinsics):
    """Return ideal pixel coordinates; inverse Brown coefficients are NOT OpenCV D."""
    points = np.asarray(points, dtype=float)
    model = intrinsics["model"]
    k = camera_matrix(intrinsics)
    if model == "distortion.none":
        return points.copy()
    if model == "distortion.brown_conrady":
        return cv2.undistortPoints(
            points.reshape(-1, 1, 2), k, np.asarray(intrinsics["coeffs"]), P=k
        ).reshape(-1, 2)
    if model == "distortion.inverse_brown_conrady":
        import pyrealsense2 as rs

        intr = rs.intrinsics()
        for key in ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs"):
            setattr(intr, key, intrinsics[key])
        intr.model = rs.distortion.inverse_brown_conrady
        rays = np.asarray(
            [rs.rs2_deproject_pixel_to_point(intr, p.tolist(), 1.0) for p in points]
        )
        return rays[:, :2] * [intr.fx, intr.fy] + [intr.ppx, intr.ppy]
    raise ValueError(f"Unsupported camera distortion: {model}")


class Markers:
    def __init__(self, config, intrinsics):
        self.c = config
        self.intrinsics = intrinsics
        self.k = camera_matrix(intrinsics)
        self.d = np.zeros(5)
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, config["dictionary"])
        )
        params = cv2.aruco.DetectorParameters()
        refinement = config.get("corner_refinement", "subpix")
        methods = {
            "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
            "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
        }
        if refinement not in methods:
            raise ValueError("Unknown corner refinement method")
        params.cornerRefinementMethod = methods[refinement]
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        s = config["size_m"] / 2
        if s <= 0:
            raise ValueError("Measure marker black-square side")
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)

    def measure(self, rgb):
        corners, ids, _ = self.detector.detectMarkers(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        )
        if ids is None:
            raise ValueError("No markers")
        centers, errors = [], []
        for marker in self.c["ids"]:
            indices = np.flatnonzero(ids.ravel() == marker)
            if len(indices) != 1:
                raise ValueError(f"Missing or duplicate marker {marker}")
            pts = corners[indices[0]].reshape(4, 2).astype(float)
            if (
                np.min(np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1))
                < self.c["min_side_px"]
            ):
                raise ValueError("Marker too small in image")
            pts = rectify_points(pts, self.intrinsics)
            _, rv, tv, _ = cv2.solvePnPGeneric(
                self.obj, pts, self.k, self.d, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            candidates = []
            for r, t in zip(rv, tv):
                r, t = cv2.solvePnPRefineLM(self.obj, pts, self.k, self.d, r, t)
                projected, _ = cv2.projectPoints(self.obj, r, t, self.k, self.d)
                error = float(
                    np.sqrt(
                        np.mean(np.sum((projected.reshape(4, 2) - pts) ** 2, axis=1))
                    )
                )
                if t[2, 0] > 0 and np.isfinite(error):
                    candidates.append((error, t.ravel()))
            candidates.sort(key=lambda v: v[0])
            if not candidates or candidates[0][0] > self.c["max_reprojection_px"]:
                raise ValueError("Marker reprojection error")
            # Ambiguity in translation affects aperture even when reprojection is good.
            if (
                len(candidates) > 1
                and candidates[1][0] < self.c["max_reprojection_px"]
                and np.linalg.norm(candidates[0][1] - candidates[1][1])
                > self.c["max_ambiguity_m"]
            ):
                raise ValueError("Ambiguous planar marker translation")
            errors.append(candidates[0][0])
            centers.append(candidates[0][1])
        distance = float(np.linalg.norm(centers[1] - centers[0]))
        return distance, max(errors), centers

    def image_separation(self, rgb):
        """Projective marker centers in raw pixels; no inferred 3D square pose.

        Valid only with an independently measured opening calibration for this
        rigid camera/jaw geometry. Missing markers are never held or invented.
        """
        corners, ids, _ = self.detector.detectMarkers(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        )
        if ids is None:
            raise ValueError("No markers")
        centers = []
        h, w = rgb.shape[:2]
        for marker in self.c["ids"]:
            matches = np.flatnonzero(ids.ravel() == marker)
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicate marker {marker}")
            p = corners[matches[0]].reshape(4, 2).astype(float)
            if (
                np.min(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1))
                < self.c["min_side_px"]
            ):
                raise ValueError(f"Marker {marker} too small in image")
            margin = self.c.get("min_border_px", 3)
            if (
                np.any(p < margin)
                or np.any(p[:, 0] > w - 1 - margin)
                or np.any(p[:, 1] > h - 1 - margin)
            ):
                raise ValueError(f"Marker {marker} too close to image border")
            # Intersection of the diagonals is the projective square center.
            hp = np.c_[p, np.ones(4)]
            center = np.cross(np.cross(hp[0], hp[2]), np.cross(hp[1], hp[3]))
            if abs(center[2]) < 1e-9:
                raise ValueError(f"Degenerate corners for marker {marker}")
            centers.append(center[:2] / center[2])
        return float(np.linalg.norm(centers[1] - centers[0])), -1.0, centers

    def measurement(self, rgb):
        if self.c.get("width_method", "metric_pnp") == "image_calibrated":
            return self.image_separation(rgb)
        return self.measure(rgb)

    def width(self, rgb):
        distance, error, centers = self.measurement(rgb)
        if self.c.get("width_method", "metric_pnp") == "image_calibrated":
            model = self.c.get("image_width_calibration")
            if model is None:
                raise ValueError(
                    "Run width-sample and fit-width for image-space opening calibration"
                )
            if list(rgb.shape[:2][::-1]) != model["resolution"]:
                raise ValueError("Resolution differs from opening calibration")
            lo, hi = model["separation_range_px"]
            if not lo <= distance <= hi:
                raise ValueError("Marker separation outside measured calibration range")
            width = model["gain_m_per_px"] * distance + model["offset_m"]
        else:
            width = self.c["width_gain"] * distance + self.c["width_offset_m"]
        if not self.c["width_range_m"][0] <= width <= self.c["width_range_m"][1]:
            raise ValueError("Aperture outside calibrated range")
        return width, error, centers
