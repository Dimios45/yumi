"""Live capture readiness, shared by terminal and browser."""

import time


class CaptureQuality:
    def __init__(self, detector):
        self.detector = detector
        self.last_image = None
        self.width = None
        self.error = "Waiting for RGB"

    def update(self, latest, now=None):
        now = time.monotonic() if now is None else now
        stamp = latest.get("rgb_arrival_s")
        if stamp != self.last_image and "rgb" in latest:
            self.last_image = stamp
            try:
                self.width = float(self.detector.width(latest["rgb"])[0])
                self.error = None
            except ValueError as e:
                self.width = None
                self.error = str(e)
        pose = latest.get("pose", {})
        reasons = []
        if stamp is None or now - stamp > 0.5:
            reasons.append("RGB stale or missing")
        if now - pose.get("arrival_s", -1e20) > 0.25:
            reasons.append("Tracker stale or missing")
        if pose.get("tracker_confidence") != 3:
            reasons.append("Tracker confidence below 3/3")
        if self.error:
            reasons.append(self.error)
        return {"ready": not reasons, "width_m": self.width, "reasons": reasons}
