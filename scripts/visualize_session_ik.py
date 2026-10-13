#!/usr/bin/env python3
"""Solve a LeRobot UMI session and export an interactive overhead YAM replay.

Run with ~/karma/.venv/bin/python (MuJoCo + the original replay IK solver).
The background is explicitly a fresh static photo, not recorded session video.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np
import replay_yam as replay


def encoder(path, width, height, fps):
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def configure_camera(renderer, cam_from_base, k):
    base_from_cam = np.linalg.inv(cam_from_base)
    near = .01
    for camera in renderer.scene.camera:
        camera.pos = base_from_cam[:3, 3]
        camera.forward = base_from_cam[:3, 2]
        camera.up = -base_from_cam[:3, 1]
        camera.frustum_near = near
        camera.frustum_far = 10.
        camera.frustum_top = k["ppy"] / k["fy"] * near
        camera.frustum_bottom = -(k["height"] - k["ppy"]) / k["fy"] * near
        camera.frustum_center = (k["width"] / 2 - k["ppx"]) / k["fx"] * near
        # MuJoCo's frustum_width stores the half-width at the near plane.
        camera.frustum_width = k["width"] / (2 * k["fx"]) * near
        camera.orthographic = 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", type=Path)
    p.add_argument("--background", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=replay.ROOT / "configs/umi.yam-collection.json")
    p.add_argument("--world-calib", type=Path, default=replay.ROOT / "configs/world_camera.json")
    p.add_argument("--opacity", type=float, default=.28)
    p.add_argument("--video-fps", type=int, default=15)
    p.add_argument("--tool-roll-deg", type=float, default=0.,
                   help="Capture TCP to YAM tool rotation about their shared approach (Z) axis")
    a = p.parse_args()
    if not 0 <= a.opacity <= 1 or a.video_fps <= 0 or 30 % a.video_fps:
        p.error("Opacity must be 0–1; video FPS must be a positive divisor of 30")
    if not (a.session / "COMPLETE").exists():
        p.error("Session has not completed")
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    world = json.loads(a.world_calib.read_text())
    bg = cv2.imread(str(a.background))
    if bg is None:
        raise ValueError("Cannot read background")
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    k = world["intrinsics"]
    if bg.shape[:2] != (k["height"], k["width"]):
        raise ValueError("Background and calibration dimensions differ")
    shutil.copyfile(a.background, a.output / "background.png")
    capture_ext = np.asarray(cfg["T_tracker_tcp"], float)
    tool_axes = replay.homogeneous(replay.Rotation.from_euler('z', a.tool_roll_deg, degrees=True).as_matrix())
    ext = capture_ext @ tool_axes
    cam = np.linalg.inv(np.asarray(world["T_table_cam"])) @ replay.table_from_base(world["base"])
    solver = replay.DecoupledIKSolver()
    model, data = solver.model, solver.data
    # A black world lets the browser composite the rendered arm with screen blending.
    model.vis.headlight.ambient[:] = .65
    model.vis.headlight.diffuse[:] = .65
    model.vis.headlight.specular[:] = .1
    for gid in range(model.ngeom):
        model.geom_rgba[gid] = [.12, .72, 1., 1.]
        model.geom_matid[gid] = -1
    renderer = mujoco.Renderer(model, height=k["height"], width=k["width"])
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
    episodes = []
    for ep in sorted((a.session / "extras").glob("episode_*")):
        meta, images, times, grid, tracker, valid = replay.load_capture(ep, 3, .025, 30)
        widths = [r["width_measurement"]["opening_m"]
                  if r.get("width_measurement", {}).get("valid") else None for r in images]
        capture = (meta, images, times, grid, tracker, valid,
                   np.array([r["device_s"] for r in images]), widths)
        samples, report = replay.solve(capture, ext, "gravity", 30, 200, .005)
        report.update(episode=ep.name, registration_verified=False,
                      tcp_source="provisional CAD config plus explicit capture-to-YAM tool-axis mapping",
                      T_tracker_capture_tcp=capture_ext.tolist(), T_capture_tcp_robot_tool=tool_axes.tolist(),
                      tool_roll_deg=a.tool_roll_deg, collision_checked=False)
        (a.output / f"{ep.name}.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(a.output / f"{ep.name}-ik.npz",
                            time_s=times, device_s=grid, valid=valid,
                            joints_rad=np.array([s["q"] for s in samples]),
                            target=np.array([s["target"] for s in samples]),
                            actual=np.array([s["actual"] for s in samples]),
                            gripper_closed=np.array([s["gripper"] for s in samples]),
                            width_m=np.array([np.nan if s["width"] is None else s["width"] for s in samples]))
        print(f"{ep.name}: {report['tracked_fraction']:.1%} tracked; rendering", flush=True)
        layer_name, movie_name = f"{ep.name}-arm.mp4", f"{ep.name}-overlay.mp4"
        layer_encoder = encoder(a.output / layer_name, k["width"], k["height"], a.video_fps)
        movie_encoder = encoder(a.output / movie_name, k["width"], k["height"], a.video_fps)
        frames = []
        try:
            for i in range(0, len(samples), 30 // a.video_fps):
                s = samples[i]
                data.qpos[:6] = s["q"]
                data.qpos[6:8] = (1 - s["gripper"]) * replay.YAM_FINGER_MAX
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, scene_option=options)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                configure_camera(renderer, cam, k)
                arm = renderer.render().copy()
                layer_encoder.stdin.write(arm.tobytes())
                # Same screen blend as the HTML canvas, at the requested low opacity.
                blended = np.clip(bg.astype(float) + a.opacity * arm * (1-bg/255.), 0, 255).astype(np.uint8)
                dots = {}
                for field, color, radius in (("target", (245, 177, 68), 5), ("actual", (65, 219, 255), 3)):
                    uv = replay.project(s[field][None, :3, 3], cam, k)
                    dots[field] = uv[0].round(2).tolist() if len(uv) else None
                    if len(uv) and s["valid"]:
                        cv2.circle(blended, tuple(uv[0].astype(int)), radius, color, 1, cv2.LINE_AA)
                cv2.rectangle(blended, (0, 0), (k["width"], 46), (17, 25, 36), -1)
                cv2.putText(blended, f"{ep.name}  {s['time']:.1f}s  |  YAM IK  |  opacity {a.opacity:.0%}",
                            (12, 18), cv2.FONT_HERSHEY_SIMPLEX, .44, (230, 240, 250), 1, cv2.LINE_AA)
                cv2.putText(blended, "STATIC PHOTO / PROVISIONAL ALIGNMENT", (12, 37),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (245, 177, 68), 1, cv2.LINE_AA)
                movie_encoder.stdin.write(blended.tobytes())
                if i == (len(samples)//2)//(30//a.video_fps)*(30//a.video_fps):
                    cv2.imwrite(str(a.output / f"{ep.name}-preview.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                frames.append(dict(time=s["time"], valid=s["valid"], status=s["status"],
                                   position_mm=s["position_error_mm"], rotation_deg=s["rotation_error_deg"], **dots))
        finally:
            for process in (layer_encoder, movie_encoder):
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("Video encoding failed")
        episodes.append({"name": ep.name, "layer": layer_name, "movie": movie_name, "report": report, "frames": frames})
    renderer.close()
    manifest = {"session": str(a.session.resolve()), "background": "background.png", "video_fps": a.video_fps,
                    "opacity": a.opacity, "world_calibration": world, "tcp_config": str(a.config.resolve()),
                    "background_source": str(a.background.resolve()), "background_is_static": True,
                    "registration_verified": False, "episodes": episodes}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    template = Path(__file__).with_name("session_ik_viewer.html").read_text()
    (a.output / "index.html").write_text(template.replace("__MANIFEST__", json.dumps(manifest)))
    print(f"Ready: {a.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
