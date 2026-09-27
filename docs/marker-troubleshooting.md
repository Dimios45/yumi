# Intermittent finger-marker visibility

The live inspection found marker 14 with poor contrast/blur and marker 13 decoded but above the 1 px 3D reprojection threshold. These are distinct failures. Changing marker size does not cure reprojection error, and detection of an ID does not establish an accurate square pose.

## Live adjustment

```bash
uv run --locked yumi preview-markers --config configs/umi.local.json
```

Open `http://192.168.0.187:8080` from the other laptop. This RGB-only preview writes no recording and runs until Ctrl+C. Outlines show decoded markers; the text lists missing IDs and whether the pair can be measured. The yellow rectangle is an inset alignment guide, not an acceptance guarantee.

- Keep IDs 13 and 14 fully visible at both closed and open travel limits. Move the markers away from the bottom edge, or adjust the camera angle before doing extrinsic calibration.
- Use flat, rigid matte labels, good white borders, and diffuse light. Avoid shiny tape over the tags. In the inspected frame the left marker (14) looked substantially more washed out than the right one.
- Aim the marker surfaces more directly toward the lens. A small tilted pad can help, but validate that it does not interfere with grasping or bend under use.
- Measure camera-to-marker distance. The [D405's published ideal operating range](https://www.realsenseai.com/product-family/d405-series/) is 7–50 cm; that depth specification is not a sharp RGB-focus boundary, but very close tags should be checked for blur.
- Keep hands away from the tags. Do not use a detector's occasional recovery as evidence that visibility is adequate during full manipulation.

## Opening calibration mode

`configs/umi.local.json` and `configs/hardware-test.json` now select `width_method: image_calibrated`. For a fixed camera and parallel jaws, this measures the separation of the two projective marker centers in raw pixels, then fits the mapping to physically measured jaw openings. This follows the image-distance calibration idea used in [FastUMI](https://github.com/zxzm-zak/FastUMI_Data), with multiple measured settings, fit-error checks, and no extrapolation. It is an alternative to individual-marker 3D PnP, not a claim that camera intrinsics or tool pose are calibrated.

Once both markers are consistently decoded, stop preview and hold a **measured** 20 mm inner gap:

```bash
uv run --locked yumi width-sample --config configs/umi.local.json \
  --opening-mm 20 --samples data/calibration/width-image.json
```

Repeat at least six distinct measured settings spanning at least 20 mm of actual travel. Do not enter 20 unless that is the physical opening. Each hold collects 60 samples, requires at least 80% accepted frames, and rejects more than 2 pixels p95–p05 separation variation.

```bash
uv run --locked yumi fit-width --samples data/calibration/width-image.json
```

The command reports `image_width_calibration`. Copy that object under `markers` in the config after reviewing it. The fit rejects maximum training residual above 1 mm. Independently validate at held-out openings and across the full range before approving calibration; small training residual does not establish generalization. The model refuses extrapolation beyond measured separations. Recalibrate after changing camera pose, tag position, resolution or jaw geometry. A fixed image-distance model is unsuitable if the camera moves relative to the gripper or fingers deform substantially.

The original `metric_pnp` mode remains available. `width_gain` and `width_offset_m` belong to that mode and must not be reused as pixel coefficients. Sample files from different modes are intentionally incompatible.

In image-calibrated mode the dataset's `marker_reprojection_px` diagnostic is **-1**, explicitly meaning not applicable: no 3D marker reprojection was used. The raw marker scale remains recorded for other geometry tasks. Neither mode fabricates a missing jaw measurement or freezes the last valid width.

## Persistent live server with terminal entry

To avoid restarting the camera between preview and measurement, run one server:

```bash
uv run --locked yumi calibrate-width-live --config configs/umi.local.json \
  --samples data/calibration/width-live.json
```

Keep its browser preview open at port 8080. In another terminal on this same machine:

```bash
uv run --locked yumi width-console
```

Hold the physically measured opening fixed, wait for READY, type its millimeter value (for example `30`) and press Enter. The server saves the median of five consecutive stable detected pairs, with frame numbers and observation age. The short observation may be at most two seconds old, allowing time to type after seeing the pair. Do not change the jaw opening between observing it and submitting its label. Unstable, stale and already-used observations are rejected. `s` prints current status; `q` exits the console without stopping the camera server.

The console accesses a control endpoint bound only to loopback port 8081, so run it on the GEM12 or via SSH to the GEM12. Only the server owns the camera. Do not run `preview-markers` or `width-sample` simultaneously. No measured opening is invented or automatically entered. Samples are compatible with `yumi fit-width --samples data/calibration/width-live.json`.
