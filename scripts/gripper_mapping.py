"""Measured YAM aperture mapping, independent of hardware and policy libraries."""

import numpy as np


class MeasuredGripperMapping:
    """Piecewise linear maps from measure_yam_gripper.py samples.

    Requested commands and measured feedback are separate coordinates. A fit
    against commands alone must not be used to interpret measured feedback.
    """

    def __init__(self, document, arm, interface):
        if document.get('arm') != arm or document.get('interface') != interface:
            raise ValueError('Gripper calibration arm/interface does not match deployment')
        points = document.get('points', [])
        if len(points) < 3:
            raise ValueError('At least three measured gripper calibration points required')
        try:
            values = np.asarray([[p['gap_mm']/1000, p['command'], p['measured']]
                                 for p in points], float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Every calibration point needs gap_mm, command, and measured feedback') from error
        values = values[np.argsort(values[:, 0])]
        if (not np.isfinite(values).all() or np.any(values[:, 0] < 0)
                or np.any(values[:, 1:] < 0) or np.any(values[:, 1:] > 1)
                or np.any(np.diff(values, axis=0) <= 0)):
            raise ValueError('Calibration gaps, commands and feedback must be finite and strictly increasing; commands/feedback must be in [0,1]')
        self.widths, self.commands, self.feedback = values.T

    @staticmethod
    def interpolate(value, source, target, label):
        value = np.asarray(value, float)
        if (not np.isfinite(value).all() or np.any(value < source[0])
                or np.any(value > source[-1])):
            raise ValueError(f'{label} outside measured gripper calibration range')
        return np.interp(value, source, target)

    def width_to_open(self, width_m):
        return self.interpolate(width_m, self.widths, self.commands, 'Requested aperture')

    def feedback_to_width(self, native_open):
        return self.interpolate(native_open, self.feedback, self.widths, 'Gripper feedback')

    def command_to_width(self, native_open):
        return self.interpolate(native_open, self.commands, self.widths, 'Simulated gripper command')
