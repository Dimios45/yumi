# Custom UMI: D405 + T265, no ROS

**Prepare calibration in one batch: [Print & Calibration Guide](CALIBRATION_PRINT_GUIDE.md)** — one A4 target sheet, physical checklist and commands.

A local, uv-managed capture and calibration toolkit with Viser preview and an official **LeRobot v3.0** exporter. It uses the T265's onboard visual–inertial pose estimate, D405 RGB/depth, and two finger ArUco markers. It does not drive robot hardware.

**Single-arm Karma/YAM workflow:** [capture, relative action export and offline robot feasibility](docs/single-arm-umi.md).
**ACT policy on the YAM arm:** [training, deployment, camera/proprioception inputs and inference commands](docs/act-relative-policy.md).
The latest three-episode batch has no valid jaw-width labels because marker 13
was not decoded. Fix that before collecting more training demonstrations. The
recorder now requires valid width as well as tracking before an episode starts.

**Status:** real D405 RGB/depth capture and Viser startup tested here; synthetic two-episode LeRobot export and full readback passed. T265 pose/IMU streaming and an overhead D435 snapshot are now tested; physical calibration remains pending. Marker scale, jaw geometry, tool extrinsics, timing, and physical tracking accuracy are not yet calibrated. Do not mark the example calibration verified until those measurements are complete. See [validation](docs/validation.md).

## Compatibility

| Component | Pin / decision | Reason |
|---|---|---|
| Capture Python | CPython 3.10 | Legacy RealSense wheel ABI |
| RealSense SDK / Python binding | librealsense 2.53.1 / `pyrealsense2==2.53.1.4623` | Includes D405 and T265; also the FastUMI recommendation |
| Capture NumPy | 1.26.4 | Conservative legacy binary environment |
| Capture OpenCV | opencv-python-headless 4.11.0.86 | Includes `cv2.aruco`; avoids competing packages overwriting `cv2` |
| Viser | 1.0.16 | Local web preview, no ROS |
| Export Python | CPython 3.10, separate uv environment | Keeps training/data dependencies away from acquisition |
| Export | LeRobot 0.4.3, NumPy 2.2.6, CPU Torch 2.7.1 | LeRobot v3; its rerun dependency requires NumPy 2 |
| Video | H.264 through LeRobot / PyAV | Tested encoding and complete readback; RGB originals remain lossless in raw captures |

**Do not upgrade the capture SDK to 2.54.1 or newer:** T265 support was removed. The 2.53.1 release says T265 is recognized but no longer manufacturer-tested. This is a compatible legacy baseline, not a promise of perfect tracking. D405 firmware observed here is **5.15.1.55**. No firmware was changed. A modern RealSense package installed elsewhere does not affect this project's uv environment; do not set `PYTHONPATH` or `LD_LIBRARY_PATH` to another SDK.

Sources: [RealSense 2.53.1 release](https://github.com/realsenseai/librealsense/releases/tag/v2.53.1), [SDK removal notes](https://github.com/realsenseai/librealsense/wiki/Release-Notes), [FastUMI](https://github.com/zxzm-zak/FastUMI_Data), [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3). See [design and research](docs/design.md).

## Installation: Ubuntu 22.04 x86_64

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if unavailable, then:

```bash
cd ~/yumi
uv sync --locked --extra dev
uv sync --locked --project exporter --python /usr/bin/python3.10
```

The capture wheel already contains the userspace RealSense implementation. ROS, Conda, CUDA, and a system-wide replacement of librealsense are unnecessary. The kernel must expose UVC video devices and your account needs camera/USB access.

Install the **pinned SDK's udev rules** once. The repository checkout already exists in `research/librealsense` on this machine. For a fresh installation:

```bash
mkdir -p research
curl -fL https://raw.githubusercontent.com/realsenseai/librealsense/v2.53.1/config/99-realsense-libusb.rules -o research/99-realsense-libusb.rules
sudo install -m 644 research/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb
```

Unplug/replug both cameras. The T265 initially appears as `03e7:2150` and boots into `8087:0b37`; **both** need permissions. Both boot and runtime access are now working in the latest hardware test. If needed, add your login to `video`/`plugdev` and log out/in. `uv` cannot grant OS USB access.

```bash
uv run yumi devices
uv run yumi snapshot --serial 352122273221 --output artifacts/handheld.png
uv run yumi detect --image artifacts/handheld.png --output artifacts/handheld-markers.png
```

The detected handheld D405 serial is `352122273221`, marker IDs **13, 14**, dictionary **DICT_4X4_50**. Two other D405s are attached. The newly connected overhead D435 is `243622071623`; T265 is `943222111495`. Always use explicit serials.

If a legacy wheel cannot access your kernel's video backend, a source build with `FORCE_RSUSB_BACKEND=ON` is an option; see [driver fallback](docs/driver-build.md). It still needs udev permissions. Do not mix SDK shared libraries from different versions.

## Calibrate, capture, convert

`configs/umi.local.json` is ignored by Git and **does not exist in a fresh checkout**. Create it before running any command with `--config`. The following uses the previously tested handheld serials and 10 mm finger markers; confirm serials with `devices` and replace them if your hardware differs. Measure each marker's outer black square and update `markers.size_m` in metres if needed.

```bash
cd ~/yumi
uv run --locked yumi devices
uv run --locked python - <<'PYCONFIG'
import json
from pathlib import Path

path = Path("configs/umi.local.json")
if path.exists():
    print(f"Keeping existing {path}; check its serials and marker settings.")
else:
    config = json.loads(Path("configs/umi.example.json").read_text())
    config["rgb_serial"] = "352122273221"
    config["tracking_serial"] = "943222111495"
    config["markers"]["size_m"] = 0.010
    config["markers"]["width_method"] = "image_calibrated"
    with path.open("x") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Created {path}; physical calibration is still required.")
PYCONFIG
```

This creates an **unverified starting config**, with no fitted jaw model or tool transform. Keep an existing calibrated config instead of overwriting it. The historical local settings mentioned in the [Print & Calibration Guide](CALIBRATION_PRINT_GUIDE.md) and troubleshooting notes are not bundled with a fresh checkout.

**Calibrate jaw opening.** Only one process can use the D405 at a time. Stop any recording or preview first. The live server requires `markers.width_method` to be `image_calibrated` (set above):

```bash
uv run --locked yumi calibrate-width-live --config configs/umi.local.json \
  --samples data/calibration/width-live.json
```

Open **http://127.0.0.1:8080**. In a second terminal on the capture computer:

```bash
cd ~/yumi
uv run --locked yumi width-console
```

Hold a physically measured inner jaw gap steady, wait for **READY**, then type its actual value in millimetres and press Enter. Collect at least six distinct measured openings spanning at least 20 mm and covering the usable travel. Set `markers.width_range_m` to your physical travel limits. Keep separate measured openings for validation. Use a new samples filename when recalibrating after a mounting or resolution change.

Type `q` to exit the console, then **Ctrl+C in the server terminal** to release the camera. Fit the samples and save the result:

```bash
uv run --locked yumi fit-width --samples data/calibration/width-live.json \
  > data/calibration/width-fit.json
```

Proceed only if the fit succeeds. It rejects maximum training error above 1 mm by default. Review the result, then install its jaw model in the local config:

```bash
uv run --locked python - <<'PYCONFIG'
import json
from pathlib import Path

path = Path("configs/umi.local.json")
config = json.loads(path.read_text())
fit = json.loads(Path("data/calibration/width-fit.json").read_text())
config["markers"]["width_method"] = fit["width_method"]
config["markers"]["image_width_calibration"] = fit["image_width_calibration"]
config["calibration_verified"] = False
path.write_text(json.dumps(config, indent=2) + "\n")
PYCONFIG
```

Validate displayed opening at held-out measured gaps using `uv run --locked yumi preview-markers --config configs/umi.local.json`; stop it before the next capture. See [marker troubleshooting](docs/marker-troubleshooting.md) for visibility and fit failures.

**Calibrate camera/tracker geometry and timing.** Print [the stationary ID 0 target](calibration/aruco_0_100mm.svg) at actual size, measure its black square, and fix it to a rigid stationary board. During the recording below, move the handheld with varied rotations about at least two axes and some translation, keeping the board visible. Let the T265 establish confidence 3 first. Replace `0.100` with the measured board side in metres:

```bash
uv run --locked yumi record --config configs/umi.local.json \
  --output data/calibration/board --seconds 60
uv run --locked yumi inspect --raw data/calibration/board \
  --config configs/umi.local.json --output data/calibration/board-audit.json
uv run --locked yumi calibrate-board --raw data/calibration/board \
  --marker-id 0 --size-m 0.100 --output data/calibration/board-result.json
```

Use new output paths for repeat recordings/results. Resolve frame-loss or timing-gate failures before accepting calibration. The solver saves `T_tracker_camera` and `camera_time_offset_s`; it does **not** update the local config. Copy the validated `camera_time_offset_s` into the config. Obtain `T_camera_tcp` from measured tool geometry/CAD and set the config's `T_tracker_tcp` to `T_tracker_camera @ T_camera_tcp` (4×4 matrix, translation in metres). The board cannot infer your tool contact point. Follow [tool geometry and validation](docs/calibration.md#4-define-the-tool-center-and-axes), including independent board captures and fixed-tip checks. Set `calibration_verified` and `timing_verified` true only after the physical spatial and timing checks pass. There is no single command that supplies all physical measurements automatically.

A camera-only diagnostic works before calibration (cannot become a tracked demonstration):

```bash
uv run yumi record --config configs/umi.local.json --output data/camera-check --seconds 15 --camera-only
```

A tracked demonstration, after completing and validating calibration:

```bash
uv run yumi record --config configs/umi.local.json --output data/raw/episode-000 --seconds 30
```

Open **http://127.0.0.1:8080** during capture. Viser binds to `0.0.0.0`; on another laptop on this LAN, open `http://192.168.0.187:8080` (or the host’s current LAN address). Viser shows RGB, marker diagnostics when scale is supplied, and latest tracker/tool pose. The preview is not the synchronized export. Capture ends at the requested duration or Ctrl+C; each command creates a new episode directory. This single-episode command restarts devices; two seconds of processing warmup does not guarantee tracker confidence. Use persistent sessions below for repeated collection. Use `--no-viewer` for capture without a browser server.

```bash
uv run yumi prepare --raw data/raw/episode-000 --calibration configs/umi.local.json --output data/prepared/episode-000.json
uv run --project exporter python exporter/export_dataset.py export \
  --root data/lerobot/my-task --repo-id local/my-task \
  --manifests data/prepared/episode-000.json
uv run --project exporter python exporter/export_dataset.py verify \
  --root data/lerobot/my-task --repo-id local/my-task
```

Pass multiple prepared JSON files to `--manifests` for multiple episodes. No automatic Hub upload occurs. Existing output paths are never overwritten. A failed conversion leaves `.partial` for inspection; choose a new destination when retrying.

A prepare failure identifies an episode requiring correction/recollection. Missing markers, tracking loss, repeated images, pose jumps, excessive width speed, clock resets, and timestamp gaps are not silently filled. Keep raw recordings; all calibration/processing is repeatable.

## Persistent episode sessions (LeRobot v3)

Use `record-session` to open the D405, T265 and Viser **once**, warm up tracking,
then start each episode from the terminal. Install the isolated official v3
writer once (its NumPy version differs from the camera SDK environment):

```bash
uv sync --project exporter --locked
uv run --locked yumi record-session \
  --config configs/umi.local.json \
  --output "data/lerobot/session-$(date +%Y%m%d-%H%M%S)" \
  --episodes 3 --seconds 30 --warmup 30 \
  --task "Pick up the block and place it in the container"
```

Use your intended calibrated or explicitly provisional config. A session never
sets calibration/timing verification flags to true. Open **http://localhost:8080**
and keep it open. During the initial 30 seconds, move gently so tracking can
initialize. Thirty seconds is a minimum warmup, not a promise of confidence.
At each prompt, press **Enter** to start; starting requires fresh RGB and fresh
tracking confidence 3/3 and valid jaw width sustained for one second. If not ready, wait and press
Enter again. Devices and the browser remain live during warmup, saving and resets;
only active episode samples are saved. Each episode ends at `--seconds`.

- **Ctrl+C while recording:** finish and save the shortened episode, then prompt
  for the next one. It counts toward `--episodes` if it contains RGB frames.
- **q + Enter, or Ctrl+C at the ready prompt:** finalize the dataset and exit.
- Tracking loss during an episode is recorded in quality fields, not silently
  removed or presented as valid. Writer/sensor failures mark the session failed.
- A conservative free-space check runs before each episode. Use a new output
  directory for every session; existing datasets are never overwritten.

The result is an **official LeRobot v3 raw-observation dataset**: `meta/`, parquet
`data/`, and RGB MP4 `videos/`. `extras/episode_*/` retains full-rate pose/IMU and
sensor timestamps plus losslessly compressed native Z16 depth (`.npz`, key
`depth`). `session.json` documents capture configuration and semantics. Detailed
encoder logs go to the adjacent `*.writer.log`, keeping the terminal readable.

This capture schema deliberately has **no `action` feature**. Its
`observation.state` is a provisional TCP pose in the persistent T265 session frame
and aperture in metres, paired with the last received pose, not a claimed
synchronized training label. `observation.quality` contains tracking confidence,
pose-valid and width-valid flags, and image-minus-pose time. Missing width uses
zero with `width_valid=0`; missing pose uses identity with `pose_valid=0`.
`observation.sensor_time` preserves real device/host times independently of
LeRobot's nominal FPS timestamps. Do not train on invalid placeholders.

Use the [single-arm UMI workflow](docs/single-arm-umi.md) for synchronized relative
action chunks, official LeRobot v3 export and offline YAM IK feasibility. Physical
calibration/timing verification and a matching policy/runtime remain necessary.
The legacy `yumi prepare` and `export_dataset.py verify` commands below target the
older raw/8D next-action pipeline, **not this session observation schema**. Use the
official LeRobot loader to inspect session datasets. `yumi prepare-session` now
constructs gated training chunks separately from capture; `yumi plan-robot` checks
one chunk offline against the pinned Karma/YAM model. `COMPLETE` means
capture finalized, not calibration or policy-readiness verification.

## Legacy raw capture and action semantics

- Raw `rgb/*.png`: original RGB, lossless. Raw `depth/*.npy`: native, unaligned `uint16` Z16. Multiply by `depth_scale_m` from metadata for metres. Depth and color intrinsics/extrinsics are recorded separately.
- `samples.jsonl`: full-rate T265 poses/confidences/velocities and accel/gyro, independent RGB and depth device timestamps, frame numbers, host monotonic arrival timestamps, timestamp domains.
- `metadata.json`: serial configuration, SDK/firmware, calibrated intrinsics, depth scale, extrinsics. `COMPLETE` means the writer closed; it does **not** mean the episode passed quality checks.
- LeRobot `observation.images.d405`: RGB MP4; `observation.state`: `[x,y,z,qx,qy,qz,qw,aperture]` in metres with an xyzw unit quaternion.
- `action`: the **next sampled absolute TCP pose and opening**, in the **first retained TCP pose's coordinate frame**. The final observation is omitted because it lacks a future target. This is a Cartesian demonstration convention, not measured robot joint commands.
- `observation.capture_time` and `observation.quality` preserve real timing and measurement diagnostics alongside LeRobot's nominal FPS timestamps.
- Dataset `extras/episode_*/`: lossless depth, full-rate pose/IMU log, metadata, and prepared manifest. Depth is deliberately **not** encoded into lossy RGB video. Extras are not loaded automatically as policy observation tensors.

This legacy 8D schema differs from the new session processor's relative 10D
action-chunk contract. Do not mix their labels or policy configurations. Neither
format includes a trained policy or validated hardware controller. Bimanual
collection and absolute external-world anchoring remain outside this single-arm
path. The D435 is not needed for the single-handheld pipeline.

## Software verification

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check src tests scripts exporter/export_dataset.py
uv run python scripts/synthetic_smoke.py --root artifacts/new-synthetic
uv run --project exporter python exporter/export_dataset.py export \
  --root artifacts/new-v3-test --repo-id local/synthetic-test \
  --manifests artifacts/new-synthetic/episode_0/prepared.json artifacts/new-synthetic/episode_1/prepared.json
```

Synthetic recordings are visibly labeled and do not establish physical accuracy. The two uv lockfiles are part of the reproducible setup.
