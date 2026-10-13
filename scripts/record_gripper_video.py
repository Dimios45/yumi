"""Operator-started D405 diagnostic video. No robot, depth, or T265 access."""

import argparse
import json
import select
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/umi.yam-collection.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--port", type=int, default=8080, help="Viser live preview port"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output or ROOT / "data" / (
        "gripper-video-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )
    if output.exists():
        parser.error(f"Output already exists: {output}")
    print(f"Handheld D405: {config['rgb_serial']}; output: {output}")
    print(
        "Stop other programs using this camera first. This recorder does not access the arm."
    )
    input("Press ENTER to start camera warmup and recording: ")

    from yumi.hardware import intrinsics, sdk

    rs = sdk()
    pipeline = rs.pipeline()
    settings = rs.config()
    settings.enable_device(config["rgb_serial"])
    width, height = config["resolution"]
    fps = config["fps"]
    settings.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    output.mkdir(parents=True, exist_ok=False)
    writer = None
    viewer = None
    started = False
    count = 0
    stop = False
    phase = "without_object"
    previous_number = None
    missing = 0
    old_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        import viser

        viewer = viser.ViserServer(host="0.0.0.0", port=args.port)
        viewer.gui.configure_theme(control_width="large")
        viewer.gui.add_markdown("## Handheld D405 recording")
        status = viewer.gui.add_markdown("Camera warming up…")
        preview = viewer.gui.add_image(
            np.zeros((height, width, 3), dtype=np.uint8), label="Live wrist camera"
        )
        viewer.gui.add_markdown(
            "Terminal controls: **o + Enter** = with object, "
            "**n + Enter** = without object, **q + Enter** or **Ctrl+C** = stop and save."
        )
        print(f"Live preview: http://localhost:{viewer.get_port()}", flush=True)
        profile = pipeline.start(settings)
        started = True
        for _ in range(30):
            if stop:
                break
            pipeline.wait_for_frames(5000)
        writer = cv2.VideoWriter(
            str(output / "wrist.avi"),
            cv2.VideoWriter_fourcc(*"FFV1"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Lossless FFV1 video encoder could not open")
        metadata = {
            "config": config,
            "rgb_intrinsics": intrinsics(profile.get_stream(rs.stream.color)),
            "serial": config["rgb_serial"],
            "created": datetime.now().astimezone().isoformat(),
            "video": "wrist.avi",
            "codec": "FFV1 lossless BGR",
            "nominal_fps": fps,
            "purpose": "Marker visibility diagnostic; no tracking or training actions",
            "timing": "AVI plays at nominal FPS; frames.jsonl preserves actual sensor and host timestamps",
            "hardware": "Only serial-selected D405 RGB; no robot connection",
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("RECORDING — phase: WITHOUT OBJECT", flush=True)
        print("Type o + ENTER when starting WITH OBJECT; n + ENTER for WITHOUT OBJECT.")
        print("Stop and save: q + ENTER or Ctrl+C.", flush=True)
        last_status = time.monotonic()
        last_preview = 0.0
        with (output / "frames.jsonl").open("x") as log:
            while not stop:
                if select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line or line.strip().lower() == "q":
                        break
                    if line.strip().lower() in ("o", "n"):
                        phase = (
                            "with_object"
                            if line.strip().lower() == "o"
                            else "without_object"
                        )
                        print(f"Phase: {phase}", flush=True)
                frame = pipeline.wait_for_frames(5000).get_color_frame()
                arrival = time.monotonic()
                number = frame.get_frame_number()
                if previous_number is not None:
                    if number <= previous_number:
                        raise RuntimeError("Camera frame number repeated or reset")
                    missing += max(0, number - previous_number - 1)
                previous_number = number
                rgb = np.asanyarray(frame.get_data())
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                log.write(
                    json.dumps(
                        {
                            "index": count,
                            "frame_number": number,
                            "device_s": frame.get_timestamp() / 1000,
                            "arrival_s": arrival,
                            "domain": str(frame.get_frame_timestamp_domain()),
                            "phase": phase,
                        }
                    )
                    + "\n"
                )
                count += 1
                # Keep browser encoding to 5 Hz; the lossless recording retains every acquired frame.
                if arrival - last_preview >= 0.2:
                    preview.image = rgb.copy()
                    status.content = (
                        f"**RECORDING — {phase.replace('_', ' ').upper()}**\n\n"
                        f"Saved frames: **{count}** · Camera frames missed: **{missing}**"
                    )
                    last_preview = arrival
                if arrival - last_status >= 5:
                    log.flush()
                    print(
                        f"Saved {count} frames | {phase} | camera frames missed: {missing}",
                        flush=True,
                    )
                    last_status = arrival
        writer.release()
        writer = None
        pipeline.stop()
        started = False
        status.content = "Recording stopped. Verifying saved video…"
        # Verify that the file really contains every frame logged before completion.
        decoded = 0
        video = cv2.VideoCapture(str(output / "wrist.avi"))
        try:
            while video.read()[0]:
                decoded += 1
        finally:
            video.release()
        if decoded != count or count == 0:
            raise RuntimeError(
                f"Video verification failed: {decoded} decoded / {count} logged frames"
            )
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "frames": count,
                    "decoded_frames": decoded,
                    "missing_camera_frames": missing,
                },
                indent=2,
            )
            + "\n"
        )
        (output / "COMPLETE").write_text(
            "Diagnostic video finalized and decoded successfully.\n"
        )
        print(
            f"SAVED: {output.resolve()}\n{count} frames; {missing} missed camera frames.",
            flush=True,
        )
    except Exception as error:
        (output / "FAILED.txt").write_text(f"{type(error).__name__}: {error}\n")
        raise
    finally:
        if writer is not None:
            writer.release()
        if started:
            pipeline.stop()
        if viewer is not None:
            viewer.stop()
        signal.signal(signal.SIGINT, old_handler)


if __name__ == "__main__":
    main()
