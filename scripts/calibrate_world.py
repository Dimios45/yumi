#!/usr/bin/env python3
"""Locate the fixed world camera relative to the table from its own depth.

Fits the table plane in one D435 depth frame. That fixes the camera height and
tilt exactly; the robot base position on the table is a measured starting guess
refined later by aligning the overlay in replay_yam.py.

    cd ~/yumi && uv run python scripts/calibrate_world.py --base-y -0.22

Table frame: origin on the table directly below the camera, z up, x along the
camera's viewing direction projected onto the table, y to the left.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parents[1]


def grab(serial, frames=30):
    pipe, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipe.start(config)
    align = rs.align(rs.stream.color)
    try:
        stack = []
        for _ in range(frames):
            stack.append(np.asanyarray(align.process(pipe.wait_for_frames())
                                       .get_depth_frame().get_data()).astype(float))
        color = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
    finally:
        pipe.stop()
    stack = np.where(np.asarray(stack) > 0, np.asarray(stack), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # pixels with no depth in any frame
        depth = np.nan_to_num(np.nanmedian(stack, axis=0))*scale
    return depth, {'fx': color.fx, 'fy': color.fy, 'ppx': color.ppx, 'ppy': color.ppy,
                   'width': color.width, 'height': color.height}


def fit_table(depth, k, tolerance=.01, iterations=500):
    v, u = np.nonzero(depth > .2)
    z = depth[v, u]
    points = np.c_[(u-k['ppx'])/k['fx']*z, (v-k['ppy'])/k['fy']*z, z]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iterations):
        a = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(a[1]-a[0], a[2]-a[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        inliers = np.abs((points-a[0]) @ (n/np.linalg.norm(n))) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    plane = points[best]
    centre = plane.mean(0)
    up = np.linalg.svd(plane-centre, full_matrices=False)[2][2]
    if up[2] > 0:  # normal must point back toward the camera
        up = -up
    height = float(-centre @ up)
    forward = np.array([0., 0., 1.])-up[2]*up
    forward /= np.linalg.norm(forward)
    cam_table = np.eye(4)
    cam_table[:3, :3] = np.c_[forward, np.cross(up, forward), up]
    cam_table[:3, 3] = -height*up
    return np.linalg.inv(cam_table), {
        'camera_height_m': height,
        'tilt_from_vertical_deg': float(np.degrees(np.arccos(abs(up[2])))),
        'inlier_fraction': float(best.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/umi.local.json')
    parser.add_argument('--output', type=Path, default=ROOT/'configs/world_camera.json')
    parser.add_argument('--arm', choices=('right', 'left'), default='right')
    parser.add_argument('--camera-height', type=float, default=.90,
                        help='Measured camera height (m); overrides the depth-fit height, tilt is kept')
    parser.add_argument('--base-x', type=float, default=0., help='Arm base x in the table frame (m)')
    parser.add_argument('--base-y', type=float, default=-.22, help='Arm base y in the table frame (m, left +)')
    parser.add_argument('--base-z', type=float, default=0., help='Arm mounting surface above the table (m)')
    parser.add_argument('--base-yaw', type=float, default=0., help='Arm +X heading from table +x (deg)')
    args = parser.parse_args()
    serial = json.loads(args.config.read_text())['world_serial']
    depth, k = grab(serial)
    table_cam, fit = fit_table(depth, k)
    if args.camera_height:
        table_cam[2, 3] = fit['measured_camera_height_m'] = args.camera_height
    result = {
        'serial': serial, 'intrinsics': k, 'T_table_cam': table_cam.round(6).tolist(), 'table_fit': fit,
        'base': {'arm': args.arm, 'x_m': args.base_x, 'y_m': args.base_y,
                 'z_m': args.base_z, 'yaw_deg': args.base_yaw},
        'note': 'Tilt from a depth plane fit; camera height and base offset are tape-measured.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), **fit, 'base': result['base']}, indent=2))


if __name__ == '__main__':
    main()
