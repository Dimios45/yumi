# Calibration and acceptance

## 1. Identify geometry before collecting demonstrations

Use `uv run yumi devices` and set exact serials. The inspected handheld is D405 `352122273221`; IDs 13/14 are detected by `DICT_4X4_50`. T265 serial is `943222111495`.

Measure each marker's **outer black square**, excluding the white border, with calipers. Both markers must have the same physical side length for the current detector. Enter metres in `markers.size_m` (your measured size is **10 mm = 0.010 m**, already entered in `configs/umi.local.json` and `configs/hardware-test.json`). Check print scaling in both directions, rigid flat mounting, a visible white margin, and focus across the full jaw range.

The current images show markers close to the bottom border and oblique to the camera. Ensure they remain fully visible through all openings and representative grasps. Increase resolution only after enumerating supported profiles and benchmarking throughput. The selected RGB and depth profiles must both support the configured resolution/FPS.

## 2. Calibrate actual jaw aperture

**Current local configuration uses image-space opening calibration.** Follow [marker troubleshooting and image calibration](marker-troubleshooting.md) for `width-image.json` and its `image_width_calibration` model. The metric-PnP steps below describe the older optional mode; do not mix the sample units or coefficients.

At least six distinct measured openings, spanning the operating range, plus separate validation openings. The helper captures 60 frames at each held setting, requires at least 80% valid measurements and less than 1 mm p95–p05 separation variation, and appends one aggregate to a JSON file:

```bash
# Hold an accurately measured 20 mm inner jaw opening with a spacer first.
# Replace 20 with the ACTUAL measured opening; do not type an assumed setting.
uv run --locked yumi width-sample --config configs/umi.local.json \
  --opening-mm 20 --samples data/calibration/width.json
```

Repeat at six distinct measured settings across your actual range (plus repeat captures), with no other process using the camera. Then:

```bash
uv run --locked yumi fit-width --samples data/calibration/width.json
```

This does not auto-approve calibration. Inspect fitting errors and validate at separate openings. On the supplied diagnostic recordings, most metric fits currently fail despite ID detection, so improve marker visibility/flatness and check camera intrinsics before trying to fit widths. A rejected capture does not append a measurement.

For individual-image diagnosis, the lower-level commands remain available:

```bash
uv run yumi snapshot --serial 352122273221 --output artifacts/width-01.png
uv run yumi measure-markers --image artifacts/width-01.png \
  --intrinsics artifacts/width-01.json --config configs/umi.local.json
```

Hold each jaw opening with a gauge or caliper. Take multiple images per setting. Put measured marker separations and actual **inner contact-surface separation** into `width-samples.json`:

```json
{
  "marker_separation_m": [0.020, 0.030, 0.040, 0.050, 0.060, 0.070],
  "jaw_opening_m":       [0.000, 0.010, 0.020, 0.030, 0.040, 0.050]
}
```

These are illustrative numbers only.

```bash
uv run yumi fit-width --samples width-samples.json
```

Copy `width_gain` and `width_offset_m` into the config. Set `width_range_m` from physical travel. Validate on withheld openings, at different handheld orientations and lighting. Report bias, RMS, maximum error and dropout rate. If aiming for 1 mm aperture accuracy, require that on held-out physical measurements; low reprojection error alone does not establish it. Do not clamp bad estimates into range.

The linear center-distance model assumes parallel jaws with fixed marker geometry. If marker centers have a substantial fixed perpendicular offset, distance becomes nonlinear with opening. Correct mounting or implement a calibrated rail model before collecting training data; do not merely increase thresholds.

## 3. T265-to-camera and timing calibration

Print the supplied [`calibration/aruco_0_100mm.svg`](../calibration/aruco_0_100mm.svg) on A4 at **100% / actual size**, with browser print margins disabled. It contains ID 0 and a 100 mm reference line. Measure its black-square side. Fix it to a rigid stationary board within the D405's usable field of view. Do not use either moving finger marker as the stationary board.

First resolve RGB frame loss and the arrival-jitter gate reported by `yumi inspect`; the board solver now refuses a capture that exceeds that gate. Record a dedicated 60-second tracked raw session. Spend the initial portion letting T265 establish tracking, then collect varied board motion once confidence stays at 3. Move the handheld around the board, keeping it visible; vary rotations about at least two axes and vary angular speed nonperiodically. Avoid fast blur, long marker occlusion, and T265 relocalization. Include static holds. The preview can operate before verification flags are true.

```bash
uv run yumi record --config configs/umi.local.json --output data/calibration/board --seconds 60
uv run yumi calibrate-board --raw data/calibration/board \
  --marker-id 0 --size-m 0.100 --output data/calibration/board-result.json
```

Replace `0.100` with the measured board size. The command preserves the shared SDK-global timebase and estimates the residual correction **added to camera timestamps**, then solves for `T_tracker_camera` using paired tracker and camera-board poses. It reports clock/correlation-related errors, hand–eye board-position and rotation residuals, and paired poses. It does not automatically set verification flags. A single planar board is susceptible to pose ambiguity; inspect residuals and validate the solution on independent captures. A calibrated multi-tag board is preferable if the single-tag solution is unstable.

A solver also accepts externally prepared synchronized/static pairs:

```bash
uv run yumi hand-eye --samples hand-eye-pairs.json
```

The JSON must contain equal-length `T_world_tracker` and `T_camera_board` arrays of 4x4 matrices, at least 12 poses with rotations about multiple axes. The board must be stationary in the tracker world.

## 4. Define the tool center and axes

Choose the tool center at the midpoint of the jaw contact region, with explicitly documented axes. Obtain `T_camera_tcp` from measured CAD/geometry, or use a properly designed fixture calibration. This repository does not infer tool-tip geometry from the two finger-marker IDs alone.

Compute:

```
T_tracker_tcp = T_tracker_camera @ T_camera_tcp
```

If you directly measure tracker-to-tool geometry, enter `T_tracker_tcp` directly. The matrix translation is in metres and its rotation must be orthonormal with determinant +1. The T265 pose origin is the SDK tracker pose frame, not an arbitrary housing corner. Account for that difference when using CAD measurements.

Test by keeping the physical tool tip on a fixed point while rotating the handheld. The reconstructed tool tip should stay within your required error. Repeat at multiple positions; measure error externally. Do not use identity as a placeholder and mark it calibrated.

## 5. Timing and trajectory acceptance

Set `camera_time_offset_s` from board calibration. Repeat the calibration with representative motion, exposure, resolution, USB topology, and machine load. Correlation peak resolution is not an uncertainty bound. Compare independent estimates and measured residual motion alignment.

Move back to known physical reference poses, perform fixed-distance straight motions, and check stationary jitter and slow drift. Confidence 3 alone does not establish millimetre accuracy. If absolute tracking accuracy is required, validate against an external calibrated reference; the two finger markers cannot correct global T265 drift.

Keep the camera warmup, then inspect frame counters/timestamps. Any duplicate frame or substantial gap causes preparation to fail. Improve illumination (shorter exposure), USB topology/power, SSD throughput, and host load before relaxing gates. This machine's initial captures had timing gaps; see `docs/validation.md`.

Only after spatial and timing validation set `calibration_verified` and `timing_verified` true. Keep a dated calibration report with marker size, jaw measurements, transforms, timing measurements, firmware, and validation statistics beside the config. Recalibrate after remounting either camera or markers.
