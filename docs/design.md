# Tracking and data design

## What is taken from the research

[UMI, RSS 2024](https://umi-gripper.github.io/) emphasizes the policy interface, relative trajectories, and latency matching. Its [pose utilities](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/common/pose_util.py) and [latency utilities](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/common/latency_util.py) motivate explicit frame composition and time calibration here. This implementation does not reproduce UMI's GoPro/ORB-SLAM reconstruction, training system, or reported performance.

[FastUMI paper](https://arxiv.org/abs/2409.19499) and [official code](https://github.com/zxzm-zak/FastUMI_Data) use T265 trajectories, a camera, and finger markers. The inspected code/config uses fixed TCP offsets and pixel-distance opening calibration with missing-marker interpolation. Here we instead require a full rigid transform, measured metric marker scale, explicit aperture calibration and rejection of missing measurements. FastUMI's ROS collection and Conda environment are not dependencies.

[RealSense 2.53.1 source](https://github.com/realsenseai/librealsense/tree/v2.53.1), commit `23b0904ba126e87327bc2908c1a5f79342eae867`, includes D405 PID `0x0b5b` and TM2/T265 support. [2.54.1 removal notes](https://github.com/realsenseai/librealsense/wiki/Release-Notes) explain why installing the latest SDK is inappropriate for this pair.

[LeRobot 0.4.3 dataset implementation](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/datasets/lerobot_dataset.py) is used directly. `create`, `add_frame`, `save_episode`, and `finalize` produce genuine v3 relational Parquet/MP4 storage. Full loader and video readback are performed before the output directory is finalized.

## Pose

T265 uses its fisheye cameras and IMU internally to estimate six-degree-of-freedom visual–inertial pose. The IMU alone cannot provide drift-free position. We save its raw motion samples for diagnostics; we do not double-integrate and fuse them again into the already-fused pose.

`T_A_B` maps coordinates in B into A. The pose composition is:

```
T_world_tcp(t) = T_world_tracker(t) @ T_tracker_tcp
T_episode_tcp(t) = inverse(T_world_tcp(t0)) @ T_world_tcp(t)
```

T265 pose axes are right-handed, x right, y up, z backward in the SDK pose convention. D405 optical axes are x right, y down, z forward. A measured full extrinsic handles their mounting; translations alone are insufficient. RealSense depth-to-color rotation arrays are stored in column-major order as returned by the SDK.

Confidence 3 is required by default. Lost tracking or physically implausible raw-pose transitions reject an episode. A slow drift can still pass: T265 confidence is not an absolute-error certificate. Relocalization can shift the world estimate, so revisit known reference poses and reject discontinuities. No absolute accuracy or drift-free guarantee is claimed.

## Finger opening

Subpixel ArUco corners and the measured black-square side produce two metric marker translations through IPPE-square PnP. Competing planar solutions are checked for translation ambiguity. D405 inverse Brown–Conrady intrinsics are rectified through RealSense deprojection, not passed incorrectly as OpenCV forward-distortion coefficients. Error is measured in rectified pixel coordinates.

Opening is `gain * ||center_right - center_left|| + offset`. The coefficients are fitted to caliper/gauge measurements. This model requires fixed, approximately coplanar marker mounts and parallel jaw travel. Different marker heights, finger flex, a scissor mechanism, or nonparallel motion can invalidate it; held-out physical measurements must establish suitability. Marker orientation is not used as a world pose anchor: both finger markers move with the handheld device.

No causal smoothing is applied to avoid adding uncalibrated phase lag. Tiny markers, oblique angles, blur, print errors, and poor lighting can dominate accuracy. Improve geometry/resolution/lighting before relaxing quality thresholds.

## Timing and failure handling

Private device epochs are never directly compared across cameras. If both devices report SDK global time, those timestamps retain one shared origin; independent arrival-based offsets must not overwrite their relative acquisition timing. For two private clocks only, device seconds are mapped onto host monotonic time with an affine fit and lower-percentile arrival offset. A separately measured camera-time correction accounts for residual exposure/timing offset. Arrival jitter measures delivery variation, not direct exposure synchronization error; it remains a conservative capture-quality gate. This is software synchronization, not hardware synchronization. Short/noisy recordings may fail the rate/jitter gate. Board motion calibration estimates timing by correlating angular-speed magnitudes and rejects weak or ambiguous peaks.

RGB is selected near a regular grid without repeating frames. Pose translation is interpolated linearly and rotation with shortest-path SLERP at the actual corrected RGB time. Extrapolation and long pose gaps are forbidden. Actual RGB/grid skew is retained. Dataset timestamps are nominal; they do not claim simultaneous hardware exposure.

Sensor polling and disk writing use separate threads with bounded buffering. A full queue aborts rather than silently dropping frames. Disk errors propagate, failed sessions retain `FAILED.txt`, and successful sessions receive `COMPLETE` after flushing. Camera USB/UVC frame losses remain possible and are caught at preparation. Raw PNG/Z16 output can be large; benchmark an SSD at the intended resolution, exposure and recording duration. The current implementation favors auditable lossless capture over minimal disk size.
