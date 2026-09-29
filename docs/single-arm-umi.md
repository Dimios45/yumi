# Single-arm UMI: implemented path and remaining gates

The capture rig is D405 RGB/depth + T265 tracking + marker-measured YAM jaw opening.
The robot target is one YAM arm controlled through Karma. Recording works without
starting the cameras again between episodes. An offline processor now builds
relative Cartesian action chunks and writes them through official LeRobot v3.
An offline YAM planner tests full-pose IK and motion limits. A matching diffusion
baseline supports training and offline prediction. **There is not yet a policy
trained on usable real demonstrations or a hardware-validated inference loop.**

## Current hardware/data status

`data/lerobot/session-20260929-222142` contains 3 episodes and 2,705 RGB frames.
The full audit found zero valid aperture samples: marker 13 was not decoded.
Confidence 3/3 does not make these action labels usable. Keep this batch for
camera/tracking diagnostics; it cannot currently produce complete UMI actions.
See `artifacts/session-review/umi-audit-current/audit.json`.

`configs/umi.cad-provisional.json` retains the corrected CAD estimate. Camera-to-tip
registration and camera/pose time offset remain provisional. The pivot test does
not independently certify those transforms or timing. Neither exporter nor
recorder changes their verification flags.

No CAN interface is present on this computer. Robot execution, camera mounting
agreement with the training view, measured gripper-command mapping and latency
matching have not been tested. Perfect tracking/IK is not a guarantee of UMI;
tracking failures and unreachable targets must be rejected.

## 1. Fix the visible aperture measurement before collecting another batch

```bash
cd ~/yumi
uv run --locked yumi preview-markers --config configs/umi.cad-provisional.json
```

Keep **http://localhost:8080** open. Marker **13 belongs on the left in the camera
image, 14 on the right**. Both complete black borders and a clear white margin
must be visible. Slowly open and close the jaws. The preview must repeatedly
report a valid width, including at the openings used by your task. If 13 never
decodes, correct its printing/visibility before continuing. The previously
measured fit covers about **0–59 mm**; do not extrapolate it to larger openings.
Changing marker placement requires repeating the width calibration.

## 2. Capture persistent episodes

```bash
uv run --locked yumi record-session \
  --config configs/umi.cad-provisional.json \
  --output "data/lerobot/session-$(date +%Y%m%d-%H%M%S)" \
  --episodes 3 --seconds 30 --warmup 30 \
  --task "Insert the batteries into the wooden holder"
```

Use the actual task description. Move gently during the initial warmup. At the
prompt, Enter starts one episode only after fresh confidence 3/3 **and valid jaw
width** remain available for one second. Ctrl+C during recording saves a shortened
episode. At the next prompt, reset the task, then Enter again; q exits. The browser
and both devices stay live throughout. Width failures during recording appear in
the terminal/browser and in quality fields; they are never turned into valid zeros.
The start gate is not a substitute for checking the complete episode.

For initial deployment, collect with the same wrist camera arrangement, tool-axis
convention and task workspace/view distribution as the robot. A handheld T265 world
origin need not equal the robot base origin. The policy contract below uses motion
relative to the current TCP; robot reachability still depends on its actual start
configuration. Do not expect arbitrary demonstrations outside the robot workspace
to become reachable through IK.

## 3. Audit and construct synchronized action chunks

Replace `SESSION` with the completed session path. Use a new output directory.

```bash
uv run --locked yumi prepare-session \
  --raw data/lerobot/SESSION \
  --calibration configs/umi.cad-provisional.json \
  --output data/prepared/SESSION --history 2 --horizon 16
```

The processor reads the full-rate sensor logs and RGB videos, maps sensor clocks,
interpolates poses at corrected RGB times, and remeasures aperture. It rejects
extrapolation, confidence loss, gaps, discontinuities and invalid widths. It never
joins chunks across rejected intervals or source episodes. Exit status 2 means
there are no usable action chunks; inspect `audit.json` for the specific reasons.

Unverified calibration blocks action construction by default. For **offline
inspection only**, add `--allow-provisional`; the manifest still says unverified.
Do not mark a real config verified simply to bypass this gate. An independently
checked TCP reference and measured temporal alignment are required before treating
these as physically calibrated training labels.

Each manifest has history 2 and horizon 16 by default:

- A pose has 10 values: xyz metres, rotation matrix column 0, column 1, aperture metres.
- All history and future poses use the **same current observation TCP** as origin.
- Future action timestamps are retained; they are not inferred from delivery time.
- Action aperture is the observed future jaw opening, not a guessed motor command.

Write usable manifests to official LeRobot v3:

```bash
uv run --locked --project exporter python exporter/export_umi.py \
  --manifests data/prepared/SESSION/episode_*.json \
  --root data/lerobot-training/SESSION --repo-id local/single-arm-umi
```

The exporter uses staging and reads every sample/image back before publishing its
output directory. Provisional manifests require `--allow-provisional` here too.
`umi-contract.json` records semantics, source hashes, calibration and source mapping.

This is a **custom action-chunk schema**, not a drop-in replacement for Karma's
standard bimanual joint-policy dataset. `observation.state` is `[history,10]`,
`action` is `[horizon,10]`. The extra `d405_history_*` video features are temporal
views of the same camera. A trainer must consume these chunks directly rather than
apply another future-action window, and use the same anchor/rotation/normalization
at inference. The matching baseline below consumes these chunks directly. A real
trained checkpoint and a live receding-horizon controller still need validation
on the arm.

## Train and inspect the matching baseline

Once you have a verified export with at least two usable source episodes:

```bash
uv run --locked --project exporter python exporter/umi_policy.py \
  --dataset data/lerobot-training/SESSION --output data/policies/umi-baseline \
  --steps 10000 --batch-size 8
uv run --locked --project exporter python exporter/predict_umi.py \
  --checkpoint data/policies/umi-baseline/policy.pt \
  --dataset data/lerobot-training/SESSION --index 0 \
  --output artifacts/predicted-chunk.json
```

This uses pinned LeRobot's ResNet18 diffusion model with a smaller temporal UNet,
two RGB observations resized to 192×256, and all-future relative 10D actions.
It deliberately bypasses LeRobot's standard past-action offset in `select_action`;
its full prediction is future target 1 through target 16. Temporal image features
are stacked as history of **one** camera. RGB preprocessing and fixed pose scales
are shared by training and inference. Default normalization covers ±1 m relative
translation and 0–100 mm aperture; out-of-range labels are rejected, never clipped.
This normalization range does not expand the physical width calibration range.
Action timing must agree with nominal FPS within 12 ms.

The last source episode is held out as a whole; overlapping chunks from it never
enter training. A final validation diffusion loss is reported separately. The
saved checkpoint is reloaded and a full prediction checked before success is
reported. Ten thousand steps is an example training budget, not a promise of task
success. The installed exporter uses CPU Torch; meaningful training will be slow
here. GPU training needs a separately tested GPU environment.

`predict_umi.py` writes an offline prediction manifest that `yumi plan-robot` can
inspect. It sends no robot commands. Predictions are converted to valid rotations
and then still require reachability, timing and physical rollout checks. Random or
undertrained predictions should fail feasibility tests. The included one-step
synthetic checkpoint is explicitly marked synthetic and is not a usable policy.

## 4. Offline robot feasibility

Karma's current model builder expects the legacy I2RT YAM layout. Pin both sources;
newer I2RT releases change paths/mounts. The local `research/` copies are ignored by
git and are model references, not an installed hardware runtime. For a fresh setup:

```bash
git clone https://github.com/sra-vjti/karma research/karma
git -C research/karma checkout b4f06f6d645755e605b6c0aec7c10af3d2c911d6
git clone https://github.com/i2rt-robotics/i2rt research/i2rt
git -C research/i2rt checkout 5d47b358bafb30c65e397f2ece506550a0db4594
uv sync --locked --extra robot --extra dev
```

Copy `configs/yam.robot.example.json` to a local profile and fill in:

- `initial_joints_rad`: the actual six-joint start configuration from robot feedback.
- `T_capture_tcp_robot_tool`: registered robot grasp-frame coordinates expressed in
  capture TCP coordinates. Identity is valid only if those origins and axes agree.
- `workspace_m`: robot-base xyz minimum and maximum bounds.
- `gripper`: measured physical jaw widths paired with native Karma commands
  (**1 = open, 0 = closed**). This differs from its recorded closedness convention.

```bash
uv run --locked --extra robot yumi plan-robot \
  --manifest data/prepared/SESSION/episode_000000.json \
  --profile configs/yam.robot.local.json --sample-index 0 \
  --output artifacts/yam-plan.json
```

This command never connects to hardware. It checks pinned clean source revisions,
converts relative actions by the registered TCP transform, solves all six pose axes,
and verifies FK position/orientation residuals, limits, speed and acceleration.
It checks enabled model collision geometry at and between targets. The base/link1
bearing overlap is explicitly excluded; the robot gripper body has collision
geometry disabled in this source model. There is no table/object/cable model.
Consequently this is limited feasibility screening, not collision certification.

The model grasp site defines the target contact frame, not a flange frame with a
similar TCP name. Each plan records revisions, residuals, tool reference correction,
limits/profile and provenance. One passing chunk does not validate a whole episode
or a running policy. The example profile has null physical measurements on purpose.

Before hardware inference, integrate the matching policy with Karma's measured
state and PositionCommand API, using latency-aware scheduling, fresh CAN feedback,
tracking-error limits and hold on interruption. Validate observed robot tool motion
against requested motion at low speed, then run a small task rollout. Those checks
cannot be completed on this laptop without the arm and valid calibration.

## Reproducible software checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --extra dev --extra robot pytest -q
uv run --locked --extra dev ruff check src tests scripts exporter/export_umi.py exporter/session_writer.py
uv run --locked python scripts/session_smoke.py --root artifacts/new-umi-smoke
uv run --locked --project exporter python exporter/export_umi.py \
  --manifests artifacts/new-umi-smoke/prepared/episode_000000.json \
  --root artifacts/new-umi-smoke/lerobot --repo-id local/synthetic-umi
uv run --locked --project exporter python exporter/umi_policy.py \
  --dataset artifacts/new-umi-smoke/lerobot \
  --output artifacts/new-umi-smoke/policy-test --steps 1 --batch-size 2 --smoke-test
uv run --locked --project exporter python exporter/test_umi_policy.py
```

Synthetic data is explicitly labeled and does not validate hardware. The fixture's
minimal source metadata exercises session preprocessing; the output is constructed
and decoded through the real official LeRobot API. Robot-model tests require the
pinned checkouts and optional MuJoCo dependency; otherwise they are skipped.

References: [UMI paper](https://arxiv.org/html/2402.10329v3),
[Karma model builder](https://github.com/SRA-VJTI/karma/blob/b4f06f6d645755e605b6c0aec7c10af3d2c911d6/src/vr_teleop_kit/ik/model.py),
[Karma inference conventions](https://github.com/SRA-VJTI/karma/blob/b4f06f6d645755e605b6c0aec7c10af3d2c911d6/docs/inference.md).
