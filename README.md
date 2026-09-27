# Custom UMI: D405 + T265, no ROS

**Prepare calibration in one batch: [Print & Calibration Guide](CALIBRATION_PRINT_GUIDE.md)** — one A4 target sheet, physical checklist and commands.

A local, uv-managed capture and calibration toolkit with Viser preview and an official **LeRobot v3.0** exporter. It uses the T265's onboard visual–inertial pose estimate, D405 RGB/depth, and two finger ArUco markers. It does not drive robot hardware.

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
cd /home/yambox/yumi
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

On this machine, `configs/umi.local.json` already contains the device serials, 10 mm marker size and image-space opening mode; see [marker troubleshooting](docs/marker-troubleshooting.md). For a new setup, copy `configs/umi.example.json` to `configs/umi.local.json`. Fill in the T265 serial and follow [the calibration procedure](docs/calibration.md). The example deliberately has null scale/extrinsics and false verification flags.

1. **Measure marker black-square size** in metres, jaw openings, and tool extrinsics.
2. **Calibrate timing** using a stationary printed board and varied rotations.
3. **Validate repeatability**, marker error across the aperture range, and timing under actual capture load.

A camera-only diagnostic works before calibration (cannot become a tracked demonstration):

```bash
uv run yumi record --config configs/umi.local.json --output data/camera-check --seconds 15 --camera-only
```

A tracked episode, after setting serials:

```bash
uv run yumi record --config configs/umi.local.json --output data/raw/episode-000 --seconds 30
```

Open **http://127.0.0.1:8080** during capture. Viser binds to `0.0.0.0`; on another laptop on this LAN, open `http://192.168.0.187:8080` (or the host’s current LAN address). Viser shows RGB, marker diagnostics when scale is supplied, and latest tracker/tool pose. The preview is not the synchronized export. Capture ends at the requested duration or Ctrl+C; each command creates a new episode directory. Keep the first two seconds for initialization. Use `--no-viewer` for capture without a browser server.

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

## Saved data and action semantics

- Raw `rgb/*.png`: original RGB, lossless. Raw `depth/*.npy`: native, unaligned `uint16` Z16. Multiply by `depth_scale_m` from metadata for metres. Depth and color intrinsics/extrinsics are recorded separately.
- `samples.jsonl`: full-rate T265 poses/confidences/velocities and accel/gyro, independent RGB and depth device timestamps, frame numbers, host monotonic arrival timestamps, timestamp domains.
- `metadata.json`: serial configuration, SDK/firmware, calibrated intrinsics, depth scale, extrinsics. `COMPLETE` means the writer closed; it does **not** mean the episode passed quality checks.
- LeRobot `observation.images.d405`: RGB MP4; `observation.state`: `[x,y,z,qx,qy,qz,qw,aperture]` in metres with an xyzw unit quaternion.
- `action`: the **next sampled absolute TCP pose and opening**, in the **first retained TCP pose's coordinate frame**. The final observation is omitted because it lacks a future target. This is a Cartesian demonstration convention, not measured robot joint commands.
- `observation.capture_time` and `observation.quality` preserve real timing and measurement diagnostics alongside LeRobot's nominal FPS timestamps.
- Dataset `extras/episode_*/`: lossless depth, full-rate pose/IMU log, metadata, and prepared manifest. Depth is deliberately **not** encoded into lossy RGB video. Extras are not loaded automatically as policy observation tensors.

This is valid LeRobot storage, but an 8D Cartesian action schema needs a matching policy configuration and robot-side calibrated Cartesian controller. Relative-action/6D-rotation adapters, bimanual collection, robot retargeting, and absolute external-world anchoring are not included. The D435 is not needed for the single-handheld pipeline.

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
