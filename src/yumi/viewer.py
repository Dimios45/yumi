import cv2
import numpy as np
import viser

from .geometry import pose_matrix, transform
from .markers import Markers


class Viewer:
    def __init__(self, config, intrinsics, port):
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.scene.set_up_direction("+y")
        self.status = self.server.gui.add_markdown("Waiting for camera / tracker")
        self.image = self.server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="D405 RGB"
        )
        self.frame = self.server.scene.add_frame(
            "/tool", axes_length=0.08, axes_radius=0.003
        )
        self.config = config
        self.markers = None
        if config.get("markers", {}).get("size_m"):
            self.markers = Markers(config["markers"], intrinsics)

    def update(self, latest):
        messages = []
        if "rgb" in latest:
            rgb = latest["rgb"]
            display = rgb.copy()
            if self.markers:
                corners, ids, _ = self.markers.detector.detectMarkers(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                )
                found = [] if ids is None else ids.ravel().tolist()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(display, corners, ids)
                h, w = rgb.shape[:2]
                cv2.rectangle(display, (12, 12), (w - 13, h - 13), (255, 180, 0), 1)
                missing = [i for i in self.config["markers"]["ids"] if i not in found]
                messages.append(f"Detected: {found}; missing required IDs: {missing}")
                try:
                    distance, error, _centers = self.markers.measurement(rgb)
                    if self.config["markers"].get("width_method") == "image_calibrated":
                        messages.append(
                            f"BOTH MARKERS VISIBLE — separation {distance:.2f} pixels (not jaw width)"
                        )
                        if self.config["markers"].get("image_width_calibration"):
                            width, _, _ = self.markers.width(rgb)
                            messages.append(
                                f"Calibrated opening: {width * 1000:.2f} mm"
                            )
                    else:
                        messages.append(
                            f"Marker separation: {distance * 1000:.1f} mm; reprojection: {error:.2f} px"
                        )
                except ValueError as e:
                    messages.append(f"MEASUREMENT INVALID: {e}")
            self.image.image = display
        if "pose" in latest:
            p = latest["pose"]
            a = pose_matrix(p["position"], p["quaternion_xyzw"])
            if self.config.get("T_tracker_tcp") is not None:
                a = a @ transform(self.config["T_tracker_tcp"])
            from scipy.spatial.transform import Rotation

            q = Rotation.from_matrix(a[:3, :3]).as_quat()
            self.frame.position = a[:3, 3]
            self.frame.wxyz = q[[3, 0, 1, 2]]
            messages.append(
                f"T265 confidence: {p['tracker_confidence']}/3 (live preview uses latest pose)"
            )
        self.status.content = "\n\n".join(messages) or "Waiting for samples"

    def close(self):
        self.server.stop()


def preview_markers(config, port=8080, seconds=None):
    """Live RGB-only alignment view; no dataset files and no tracker required."""
    import time

    from .hardware import intrinsics, sdk

    rs = sdk()
    pipe = rs.pipeline()
    c = rs.config()
    c.enable_device(config["rgb_serial"])
    w, h = config["resolution"]
    c.enable_stream(rs.stream.color, w, h, rs.format.rgb8, config["fps"])
    profile = pipe.start(c)
    viewer = None
    try:
        viewer = Viewer(config, intrinsics(profile.get_stream(rs.stream.color)), port)
        end = time.monotonic() + seconds if seconds is not None else float("inf")
        while time.monotonic() < end:
            frame = pipe.wait_for_frames(3000).get_color_frame()
            viewer.update({"rgb": np.asanyarray(frame.get_data())})
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        if viewer:
            viewer.close()
