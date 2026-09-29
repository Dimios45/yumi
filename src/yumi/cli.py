import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="ROS-free T265 + D405 capture")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("devices")
    s = sub.add_parser(
        "plan-robot",
        help="Offline YAM IK/feasibility for one UMI chunk; never moves hardware",
    )
    s.add_argument("--manifest", required=True)
    s.add_argument("--profile", required=True)
    s.add_argument("--karma-root", default="research/karma")
    s.add_argument("--i2rt-root", default="research/i2rt")
    s.add_argument("--sample-index", type=int, default=0)
    s.add_argument("--output", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--serial", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("record")
    s.add_argument("--config", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--no-viewer", action="store_true")
    s.add_argument("--camera-only", action="store_true")
    s.add_argument("--port", type=int, default=8080)
    s = sub.add_parser(
        "record-session", help="Keep devices running; save prompted LeRobot v3 episodes"
    )
    s.add_argument("--config", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--episodes", type=int, default=3)
    s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--warmup", type=float, default=30)
    s.add_argument("--task", required=True)
    s.add_argument("--repo-id", default="local/yumi-session")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--no-viewer", action="store_true")
    s = sub.add_parser(
        "prepare-session",
        help="Audit and synchronize LeRobot session into relative UMI chunks",
    )
    s.add_argument("--raw", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--calibration")
    s.add_argument("--history", type=int, default=2)
    s.add_argument("--horizon", type=int, default=16)
    s.add_argument("--allow-provisional", action="store_true")
    s.add_argument("--task")
    s = sub.add_parser("prepare")
    s.add_argument("--raw", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--calibration")
    s = sub.add_parser("detect")
    s.add_argument("--image", required=True)
    s.add_argument("--dictionary", default="DICT_4X4_50")
    s.add_argument("--output")
    s = sub.add_parser("fit-width")
    s.add_argument("--samples", required=True)
    s.add_argument("--max-error-mm", type=float, default=1.0)
    s = sub.add_parser("hand-eye")
    s.add_argument("--samples", required=True)
    s = sub.add_parser("calibrate-board")
    s.add_argument("--raw", required=True)
    s.add_argument("--marker-id", type=int, required=True)
    s.add_argument("--size-m", type=float, required=True)
    s.add_argument("--dictionary", default="DICT_4X4_50")
    s.add_argument("--output", required=True)
    s = sub.add_parser("measure-markers")
    s.add_argument("--image", required=True)
    s.add_argument("--intrinsics", required=True)
    s.add_argument("--config", required=True)
    s = sub.add_parser("inspect")
    s.add_argument("--raw", required=True)
    s.add_argument("--config")
    s.add_argument("--output")
    s = sub.add_parser("width-sample")
    s.add_argument("--config", required=True)
    s.add_argument("--opening-mm", type=float, required=True)
    s.add_argument("--samples", required=True)
    s.add_argument("--frames", type=int, default=60)
    s = sub.add_parser("preview-markers")
    s.add_argument("--config", required=True)
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--seconds", type=float)
    s = sub.add_parser("calibrate-width-live")
    s.add_argument("--config", required=True)
    s.add_argument("--samples", required=True)
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--control-port", type=int, default=8081)
    s = sub.add_parser("width-console")
    s.add_argument("--control-port", type=int, default=8081)
    a = p.parse_args()
    try:
        if a.command == "plan-robot":
            from .robot_plan import make_plan

            if Path(a.output).exists():
                raise FileExistsError("Use a new plan output path")
            plan = make_plan(
                a.manifest, a.profile, a.karma_root, a.i2rt_root, a.sample_index
            )
            Path(a.output).parent.mkdir(parents=True, exist_ok=True)
            Path(a.output).write_text(json.dumps(plan, indent=2) + "\n")
            print(f"Offline plan: {a.output}; hardware execution disabled")
        elif a.command == "devices":
            from .hardware import devices

            print(json.dumps(devices(), indent=2))
        elif a.command == "calibrate-width-live":
            from .live_calibration import serve

            serve(
                json.loads(Path(a.config).read_text()),
                a.samples,
                a.port,
                a.control_port,
            )
        elif a.command == "width-console":
            from .live_calibration import console

            console(a.control_port)
        elif a.command == "preview-markers":
            from .viewer import preview_markers

            preview_markers(json.loads(Path(a.config).read_text()), a.port, a.seconds)
        elif a.command == "width-sample":
            from .calibration import capture_width_sample

            c = json.loads(Path(a.config).read_text())
            print(
                json.dumps(
                    capture_width_sample(c, a.opening_mm, a.samples, a.frames), indent=2
                )
            )
        elif a.command == "inspect":
            from .diagnostics import inspect_recording

            c = json.loads(Path(a.config).read_text()) if a.config else None
            result = inspect_recording(a.raw, c)
            if a.output:
                Path(a.output).parent.mkdir(parents=True, exist_ok=True)
                Path(a.output).write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        elif a.command == "snapshot":
            from .hardware import snapshot

            snapshot(a.serial, a.output)
        elif a.command == "record":
            from .hardware import record

            print(
                record(
                    json.loads(Path(a.config).read_text()),
                    a.output,
                    a.seconds,
                    not a.no_viewer,
                    a.port,
                    not a.camera_only,
                )
            )
        elif a.command == "record-session":
            from .hardware import record
            from .session import Session

            c = json.loads(Path(a.config).read_text())
            c["task"] = a.task
            session = Session(
                a.output, a.episodes, a.seconds, a.warmup, a.task, a.repo_id
            )
            print(
                record(c, a.output, a.seconds, not a.no_viewer, a.port, True, session)
            )
        elif a.command == "prepare-session":
            from .session_processing import prepare_session

            c = json.loads(Path(a.calibration).read_text()) if a.calibration else None
            result = prepare_session(
                a.raw, a.output, c, a.history, a.horizon, a.allow_provisional, a.task
            )
            print(json.dumps(result, indent=2))
            if not result["total_chunks"]:
                p.exit(
                    2,
                    "No usable training chunks; inspect audit.json. Raw data preserved.\n",
                )
        elif a.command == "prepare":
            from .processing import prepare

            calibration = (
                json.loads(Path(a.calibration).read_text()) if a.calibration else None
            )
            print(json.dumps(prepare(a.raw, a.output, calibration), indent=2))
        elif a.command == "detect":
            import cv2

            image = cv2.imread(a.image)
            if image is None:
                raise ValueError("Cannot read image")
            detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dictionary))
            )
            corners, ids, _ = detector.detectMarkers(image)
            print(
                json.dumps(
                    {
                        "dictionary": a.dictionary,
                        "ids": ids.ravel().tolist() if ids is not None else [],
                    }
                )
            )
            if a.output:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                if not cv2.imwrite(a.output, image):
                    raise OSError(a.output)
        elif a.command == "measure-markers":
            import cv2

            from .markers import Markers

            c = json.loads(Path(a.config).read_text())
            intr = json.loads(Path(a.intrinsics).read_text())
            intr = intr.get("rgb_intrinsics", intr)
            detector = Markers(c["markers"], intr)
            im = cv2.imread(a.image)
            if im is None:
                raise ValueError("Cannot read image")
            distance, error, centers = detector.measure(
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            )
            print(
                json.dumps(
                    {
                        "marker_separation_m": distance,
                        "reprojection_px": error,
                        "centers_camera_m": [x.tolist() for x in centers],
                    },
                    indent=2,
                )
            )
        elif a.command == "calibrate-board":
            from .calibration import calibrate_board

            result = calibrate_board(a.raw, a.marker_id, a.size_m, a.dictionary)
            with Path(a.output).open("x") as f:
                json.dump(result, f, indent=2, allow_nan=False)
            print(
                json.dumps(
                    {k: v for k, v in result.items() if not k.startswith("T_")},
                    indent=2,
                )
            )
        else:
            from .calibration import fit_image_width, fit_width, hand_eye

            samples = json.loads(Path(a.samples).read_text())
            result = (
                (
                    fit_image_width(samples, max_error_m=a.max_error_mm / 1000)
                    if "marker_separation_px" in samples
                    else fit_width(
                        samples["marker_separation_m"], samples["jaw_opening_m"]
                    )
                )
                if a.command == "fit-width"
                else hand_eye(samples["T_world_tracker"], samples["T_camera_board"])
            )
            print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError) as e:
        p.exit(1, f"Error: {e}\n")
