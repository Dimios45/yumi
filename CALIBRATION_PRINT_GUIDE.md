# Custom UMI — Print & Calibration Guide

**Hardware:** D405 wrist camera + T265 tracker + two finger markers.  
**Software:** uv, RealSense 2.53.1, Viser, LeRobot v3; no ROS.  
**Prepared:** 28 September 2026.

## 1. Print everything in one batch

### The sheet to print

**[Open PRINT_ALL_A4.svg](calibration/PRINT_ALL_A4.svg)**

This is one A4 page containing every paper target required by the **currently implemented single-handheld calibration workflow**:

| Item on sheet | Dictionary / ID | Black-square size | Purpose |
|---|---|---:|---|
| Large stationary target | `DICT_4X4_50`, ID **0** | **100 × 100 mm** | D405-to-T265 transform and residual timing calibration |
| Spare finger label | `DICT_4X4_50`, ID **13** | **10 × 10 mm** | Optional replacement for one existing finger label |
| Spare finger label | `DICT_4X4_50`, ID **14** | **10 × 10 mm** | Optional replacement for the other finger label |
| Reference line | Not a marker | **100 mm** | Check printer scaling |

**Print two identical copies in this batch:** one working sheet and one spare. Only one large board is needed during calibration. Your current finger labels are already installed; keep them if they work. The small printed tags are spares, not a requirement to replace the current pair.

A separate board-only file is available: [aruco_0_100mm.svg](calibration/aruco_0_100mm.svg).

### Printer settings

- [ ] A4 paper, portrait, **100% / Actual size**.
- [ ] Disable **Fit to page**, **Shrink to fit**, browser headers/footers and automatic scaling.
- [ ] Use black ink on white matte paper; highest practical print quality.
- [ ] Check preview: all targets and the reference line must be present, with no clipping.
- [ ] Measure the printed **black-square width AND height** with a ruler/caliper.
- [ ] Measure the reference line: it should be 100 mm.
- [ ] If printing changes the scale, correct the print settings and reprint. Do not assume the nominal size.

**Do not print a screenshot or resize the target by dragging it in a document.** Open the SVG directly in a browser or a vector-capable application. Print the target sheet separately from this Markdown guide: printing a Markdown preview may rescale embedded pictures.

### Physical assembly

- [ ] Attach the large target to a **flat, rigid board** using glue or tape behind the paper.
- [ ] Keep at least **10 mm of white space** around its black square.
- [ ] Avoid creases, bubbles and glossy tape/lamination across the target face.
- [ ] Keep the large target stationary throughout each calibration recording.
- [ ] If replacing finger labels, retain a white margin of approximately **2–3 mm** around each black square.
- [ ] Keep **only one visible copy of ID 13 and one of ID 14** on the gripper. Cover/remove obsolete copies.
- [ ] Mount finger labels rigidly, facing the wrist camera sufficiently to remain readable across the entire jaw travel.

Sideways/rotated labels are supported. Poor visibility, bending and occlusion are not solved by changing their IDs.

## 2. Prepare these non-printed items

| Item | Why it is needed |
|---|---|
| Calipers or an accurately marked ruler | Measure printed targets and actual inner jaw openings |
| Rigid flat board and stand/clamp | Hold the large target fixed |
| Matte adhesive / tape used behind labels | Prevent reflections and changing marker geometry |
| Diffuse lighting | Keep tags readable without glare or motion blur |
| Known-width spacers or gauge blocks, if available | Hold repeatable jaw openings while entering measurements |
| CAD or measured mounting geometry | Locate the gripper tool center relative to the D405 |
| Stable USB connections and sufficient recording space | Avoid interrupted or dropped sensor data |

**No mandatory 3D-printed calibration fixture or STL is required by the current workflow.** Printed paper rulers and nominal 3D-printed spacer sizes are not precision references until physically measured.

A checkerboard/ChArUco board is **not required by the currently implemented commands**. We currently use factory D405 intrinsics; a dedicated intrinsic-calibration workflow would be a separate addition if its accuracy proves insufficient. The D435 is optional for overhead inspection and is not required for this single-handheld calibration.

## 3. Current status — what is already done

| Component | Status |
|---|---|
| RealSense devices and uv installation | Working |
| Live preview accessible from another laptop | Working on port 8080 |
| Finger IDs | 13 and 14, `DICT_4X4_50` |
| Marker size in config | 10 mm; remeasure if the new labels differ |
| Jaw-opening fit | Applied **provisionally** at the user's request |
| Jaw fit errors | RMS **2.02 mm**, maximum **3.15 mm** on six fitting samples |
| User check of displayed opening | Reported accurate; no numerical held-out measurements recorded |
| D405-to-T265 calibration | Still required |
| Camera-to-tool geometry | Still required |
| Residual timing calibration | Still required |
| Camera frame-loss / arrival-jitter acceptance | Still unresolved in the audited recordings |
| Physical end-to-end accuracy | Not yet established |

The accepted **3.5 mm fit-error limit** is a provisional testing setting, not a claim of 1 mm accuracy. A valid LeRobot file does not by itself certify tracking accuracy.

## 4. Confirm the setup before calibration

```bash
cd ~/yumi
uv run --locked yumi devices
```

| Role | Serial |
|---|---|
| Handheld D405 | `352122273221` |
| T265 | `943222111495` |
| Overhead D435 | `243622071623` |

Use **`configs/umi.local.json`**. Do not overwrite it with the example config: the local file contains the applied width fit and current marker settings.

**One process must own the D405 at a time.** The live width server may still be running. Typing `q` in `width-console` closes only that console, not the camera server. Stop the server in its owning terminal with Ctrl+C—or ask the assistant to stop its running server—before starting another camera command.

## 5. Jaw-opening calibration — only repeat if needed

Keep the existing provisional fit while validating the remaining pipeline. Recalibrate opening after changing the camera mounting, marker positions, resolution or finger geometry.

If repeating, use a **new sample filename**:

```bash
uv run --locked yumi calibrate-width-live \
  --config configs/umi.local.json \
  --samples data/calibration/width-new-session.json
```

Open `http://192.168.0.187:8080` from the other laptop. In another terminal **on the GEM12**:

```bash
cd ~/yumi
uv run --locked yumi width-console
```

1. Hold a measured **inner contact-surface gap** steady.
2. Wait for **READY** in the browser.
3. Enter the actual millimeter value and press Enter.
4. Confirm **SAVED**. Each accepted entry is immediately written to disk.
5. Collect at least six distinct openings across the usable range, preferably 8–10, with several repeats.
6. Keep separate openings for independent validation; do not add every validation point to the fit.

The live sampler uses five stable detected pairs no more than two seconds old. Keep the opening unchanged while typing. It rejects missing/stale observations; it does not invent the width when a marker disappears.

Fit a new session with the default 1 mm training-residual limit:

```bash
uv run --locked yumi fit-width \
  --samples data/calibration/width-new-session.json
```

To reproduce the **current explicitly provisional** fit:

```bash
uv run --locked yumi fit-width \
  --samples data/calibration/width-live-newmarkers.json \
  --max-error-mm 3.5
```

`fit-width` prints results; it does not automatically install a new model into the config. The current model is already installed.

## 6. Stationary-board camera/tracker and timing calibration

### Physical setup

- [ ] Measure ID 0's black-square side and record it below.
- [ ] Fix the board to the table/wall/stand so it cannot move.
- [ ] Keep the whole square and white margin visible to the **D405**, not just the overhead D435.
- [ ] Keep the T265 fisheye lenses clear and facing a well-lit, textured scene.
- [ ] Keep both cameras rigidly attached to the gripper.

### Capture

After stopping the width server:

```bash
cd ~/yumi
UMI_CALIB_DIR="data/calibration/board-$(date +%Y%m%d-%H%M%S)"
uv run --locked yumi record \
  --config configs/umi.local.json \
  --output "$UMI_CALIB_DIR" \
  --seconds 60
```

During capture:

1. Allow T265 to establish tracking; look for sustained confidence **3/3**.
2. Keep the stationary target fully visible while moving the handheld.
3. Rotate gently about at least **two different axes**, with some translation.
4. Vary the speed naturally rather than making a perfectly repetitive motion.
5. Include short still holds at different orientations.
6. Avoid blur, board motion, marker occlusion and touching the camera mounts.

The board must move **in the image because the camera moves**; do not move the board to simulate this.

### Inspect and solve

Run these in the same shell so `UMI_CALIB_DIR` still names the recording:

```bash
uv run --locked yumi inspect \
  --raw "$UMI_CALIB_DIR" \
  --config configs/umi.local.json \
  --output "$UMI_CALIB_DIR/audit.json"

uv run --locked yumi calibrate-board \
  --raw "$UMI_CALIB_DIR" \
  --marker-id 0 \
  --size-m 0.100 \
  --output "$UMI_CALIB_DIR/board-result.json"
```

**Replace `0.100` with the measured black-square side in metres.** If the square measures 99 mm, use `0.099`; if width and height differ meaningfully, fix the print instead of treating a rectangle as a square.

The solver returns `T_tracker_camera`, the residual `camera_time_offset_s`, correlation diagnostics and board-pose residuals. It does not automatically approve or install calibration.

**Known blocker:** previous captures had camera frame loss and excessive delivery jitter. The board solver rejects recordings exceeding the timing gate. Printing the target does not fix this acquisition problem; resolve it before accepting the calibration. Repeat on an independent board recording to check consistency.

## 7. Camera-to-tool geometry — measurement, not another print

Define the tool center as the midpoint of the intended jaw contact region, with explicitly defined tool axes. Provide:

- CAD assembly or measured D405-to-tool translation in millimeters.
- Camera mounting orientation relative to the chosen tool axes.
- The exact contact point/region used as the tool center.
- Whether the midpoint shifts as the jaws open or the fingers flex.

The transformation convention is: **`T_A_B` maps coordinates from frame B into frame A**.

```text
T_tracker_tcp = T_tracker_camera @ T_camera_tcp
```

The config requires a complete 4×4 `T_tracker_tcp`, with translation in **metres**. Do not use the camera housing corner as the optical origin or enter identity as a placeholder for a verified tool transform.

A board can calibrate camera-to-tracker geometry; it cannot automatically identify your custom finger contact point. CAD/physical measurement or a designed fixture procedure is still needed.

## 8. Validate before collecting training episodes

- [ ] Compare displayed jaw opening against independent measured gaps.
- [ ] Check repeated returns to the same physical tool pose.
- [ ] Keep the physical tool tip fixed while rotating the handheld; assess apparent tip motion.
- [ ] Compare known physical translations against reconstructed translations.
- [ ] Check sustained tracking confidence, camera counters, timestamps and marker visibility during realistic manipulation.
- [ ] Repeat timing/board calibration independently and compare results.
- [ ] Record actual measured errors; confidence and reprojection error alone are not physical accuracy certificates.

Only after completing these checks should `calibration_verified` and `timing_verified` be set true. The width-only provisional approval does not supply missing tool geometry or timing.

## 9. Record and verify LeRobot v3 once calibration is complete

After stopping any other camera server:

```bash
cd ~/yumi
UMI_EPISODE_DIR="data/raw/episode-$(date +%Y%m%d-%H%M%S)"
uv run --locked yumi record \
  --config configs/umi.local.json \
  --output "$UMI_EPISODE_DIR" --seconds 30

uv run --locked yumi prepare \
  --raw "$UMI_EPISODE_DIR" \
  --calibration configs/umi.local.json \
  --output "$UMI_EPISODE_DIR/prepared.json"

UMI_DATASET_DIR="data/lerobot/validation-$(date +%Y%m%d-%H%M%S)"
uv run --locked --project exporter python exporter/export_dataset.py export \
  --root "$UMI_DATASET_DIR" --repo-id local/umi-validation \
  --manifests "$UMI_EPISODE_DIR/prepared.json"

uv run --locked --project exporter python exporter/export_dataset.py verify \
  --root "$UMI_DATASET_DIR" --repo-id local/umi-validation
```

The exporter performs official LeRobot readback and video verification. Preparation still rejects missing spatial/timing calibration and failed quality checks. Use new output paths; existing recordings are not overwritten.

## 10. Fill-in preparation sheet

| Measurement / information | Your value |
|---|---|
| Printed ID 0 black-square width × height | ___ mm × ___ mm |
| Printed 100 mm reference-line length | ___ mm |
| Installed ID 13 black-square width × height | ___ mm × ___ mm |
| Installed ID 14 black-square width × height | ___ mm × ___ mm |
| Maximum inner jaw opening | ___ mm |
| Camera-to-marker distance | ___ mm |
| Tool-center location / CAD file | ___ |
| Mounting rigid, old duplicate tags covered | Yes / No |
| Board fixed and matte | Yes / No |
| Independent opening-check results | ___ |
| Independent tool-pose-check results | ___ |
| Board capture and calibration-result paths | ___ |

## Files and further detail

- [One-page print sheet](calibration/PRINT_ALL_A4.svg)
- [Current configuration](configs/umi.local.json)
- [Current provisional width fit](data/calibration/width-provisional-fit.json)
- [Marker troubleshooting](docs/marker-troubleshooting.md)
- [Audits of your diagnostic attempts](docs/attempts-20260928.md)
- [Research-backed data-quality requirements](docs/research-quality-checklist.md)
- [Installation and dataset schema](README.md)

**Prepare now:** print the two copies, mount the large board, measure the targets, gather calipers/spacers, and locate the camera/tool CAD dimensions. No additional paper target is required for the currently implemented workflow.
