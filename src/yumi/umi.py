"""One explicit UMI contract shared by preprocessing and deployment.

10D poses: metres xyz, first two rotation columns, opening in metres.
Every history/action pose is relative to ONE current observation TCP.
"""

import numpy as np

from .geometry import transform

POSE_NAMES = [
    "x_m",
    "y_m",
    "z_m",
    "r00",
    "r10",
    "r20",
    "r01",
    "r11",
    "r21",
    "aperture_m",
]
CONTRACT = "yumi.umi.relative_tcp_10d.v1"


def encode_pose(matrix, width):
    a = transform(matrix)
    if not np.isfinite(width) or width < 0:
        raise ValueError("Opening must be finite and nonnegative")
    return np.r_[a[:3, 3], a[:3, 0], a[:3, 1], width]


def decode_pose(vector):
    v = np.asarray(vector, dtype=float)
    if v.shape != (10,) or not np.isfinite(v).all() or v[-1] < 0:
        raise ValueError("Expected finite 10D relative pose and nonnegative aperture")
    x = v[3:6]
    y = v[6:9]
    if np.linalg.norm(x) < 1e-6:
        raise ValueError("Degenerate rotation 6D first column")
    x = x / np.linalg.norm(x)
    y = y - x * np.dot(x, y)
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("Degenerate rotation 6D collinear columns")
    y = y / np.linalg.norm(y)
    a = np.eye(4)
    a[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    a[:3, 3] = v[:3]
    return transform(a), float(v[-1])


def make_chunks(frames, history=2, horizon=16, fps=30):
    if history < 1 or horizon < 1 or fps <= 0:
        raise ValueError("Positive history, horizon and FPS required")
    chunks = []
    for i in range(history - 1, len(frames) - horizon):
        window = frames[i - history + 1 : i + horizon + 1]
        if not all(f["valid"] for f in window):
            continue
        dt = np.diff([f["time_s"] for f in window])
        if np.any(dt <= 0) or np.any(dt > 1.5 / fps):
            continue
        anchor = transform(frames[i]["T_world_tcp"])
        inv = np.linalg.inv(anchor)

        def encode(f, inv=inv):
            return encode_pose(
                inv @ transform(f["T_world_tcp"]), f["opening_m"]
            ).tolist()

        chunks.append(
            {
                "anchor_frame": frames[i]["frame_index"],
                "history_frames": [f["frame_index"] for f in window[:history]],
                "future_frames": [f["frame_index"] for f in window[history:]],
                "observation_state": [encode(f) for f in window[:history]],
                "action": [encode(f) for f in window[history:]],
                "action_time_offsets_s": [
                    f["time_s"] - frames[i]["time_s"] for f in window[history:]
                ],
            }
        )
    return chunks


def robot_targets(anchor_robot_tool, relative_actions, T_capture_tcp_robot_tool):
    """Convert capture-axis relative chunks to robot tool targets by conjugation.

    X=T_capture_tcp_robot_tool maps robot-tool coordinates into capture TCP.
    Camera mounting/task distribution must separately match training.
    """
    anchor = transform(anchor_robot_tool)
    x = transform(T_capture_tcp_robot_tool)
    out = []
    for vector in relative_actions:
        delta, width = decode_pose(vector)
        out.append((transform(anchor @ np.linalg.inv(x) @ delta @ x), width))
    return out


def width_to_command(width, widths_m, commands, convention="native_open"):
    widths = np.asarray(widths_m, float)
    cmd = np.asarray(commands, float)
    if (
        len(widths) < 2
        or widths.shape != cmd.shape
        or not np.isfinite([widths, cmd]).all()
        or np.any(np.diff(widths) <= 0)
        or np.any(np.diff(cmd) <= 0)
        or np.any(cmd < 0)
        or np.any(cmd > 1)
    ):
        raise ValueError(
            "Need measured increasing widths and native-open commands in [0,1]"
        )
    if not np.isfinite(width) or not widths[0] <= width <= widths[-1]:
        raise ValueError("Requested opening outside measured robot gripper range")
    value = float(np.interp(width, widths, cmd))
    if convention == "native_open":
        return value
    if convention == "dataset_closed":
        return 1 - value
    raise ValueError("Unknown gripper command convention")
