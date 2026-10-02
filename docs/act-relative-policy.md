# ACT relative-trajectory policy on YAM

This is the first policy trained on real UMI demonstrations from this rig that has
run on the YAM arm. It is an LeRobot ACT model trained on handheld T265 + D405
captures. At deployment it drives the **right** YAM arm through Karma, with the
right-wrist D405 as its only camera.

| | |
|---|---|
| Task | "pick up the green block and place it on the wooden board" |
| Dataset | `vruga/yumi-umi-block-place`, local copy `data/lerobot/yumi-umi-block-place` |
| Checkpoint | `vruga/yumi-umi-block-place-act-relative`, local copy `data/policies/yumi-umi-block-place-act-relative` |
| Inference script | [`scripts/infer_act_relative.py`](../scripts/infer_act_relative.py) |
| Export script | [`scripts/export_umi_lerobot.py`](../scripts/export_umi_lerobot.py) |
| Training script | `train_act_relative.py`, shipped inside the checkpoint folder |
| Status (2026-10-03) | Sim dry run and several short live rollouts on the right arm. The operator judged them decent. No success-rate measurement yet. |

The design follows the UMI policy interface (Chi et al., *Universal Manipulation
Interface*). Comments in the code cite its sections:

- **PD1.1**: latency matching. Proprioception is interpolated to the camera frame's capture time.
- **PD1.2**: action steps that are already late when they would execute are dropped.
- **PD2.1**: actions are a trajectory relative to the current end-effector pose.
- **PD2.2**: proprioception is a relative pose history, not an absolute pose.

---

## 1. Data flow

```
                 TRAINING (handheld UMI gripper)                 DEPLOYMENT (YAM right arm)
 ┌──────────────────────────────────────────────┐   ┌──────────────────────────────────────────────┐
 │ D405 RGB 640x480 @30 ─┐                        │   │ right-wrist D405 RGB 640x480 @30 ─┐          │
 │ T265 6-DoF pose ──────┼─ interp at image time ─┤   │ YAM joints ── FK (tool0) ─────────┼─ interp  │
 │ ArUco 13/14 → width ──┘   (export_umi_lerobot) │   │ gripper pos × 95 mm ──────────────┘ at image │
 │                                                │   │                                     time     │
 │ observation.state (11) ◄── relative history    │   │ observation.state (11) ◄── same formula      │
 │ action chunk (50×10)   ◄── relative to now     │   │ ACT ──► 50×10 relative chunk                 │
 │                     ACT, 20k steps             │   │   T_now @ chunk ──► tool0 targets            │
 └──────────────────────────────────────────────┘   │   EEFollower IK 200 Hz ──► Karma clamps ──► CAN│
                                                      └──────────────────────────────────────────────┘
```

---

## 2. Data collection and export

### Capture

- **Source:** raw handheld episodes in `data/raw/session-02`. Capture and calibration are covered in [single-arm-umi.md](single-arm-umi.md) and the [README](../README.md).
- **Episodes kept:** 22, `episode-004` to `episode-025`. `episode-000` to `-003` were skipped by request.
- **Size:** 6,975 frames at 30 fps, about 3.9 minutes of demonstration.
- **Camera:** the handheld D405, SDK serial `352122273221`. **This is not the camera used at deployment** (see §5).

### Export

```bash
cd ~/karma && uv run python ../yumi/scripts/export_umi_lerobot.py \
    ../yumi/data/raw/session-02 --repo-id vruga/yumi-umi-block-place \
    --task "pick up the green block and place it on the wooden board" \
    --skip episode-000 episode-001 episode-002 episode-003 --push
```

The export runs in Karma's environment, which has LeRobot 0.6.1. Its provenance is
recorded in `meta/umi_export.json` and a copy of the script in `meta/export_umi_lerobot.py`.

How the export builds each frame:

1. **Time reference.** Each wrist RGB frame defines time `t`. The T265 pose is interpolated to `t`: linear for position, slerp for rotation. A frame is kept only if the poses on both sides of it have tracker confidence ≥ 3 and are ≤ 25 ms apart. `camera_time_offset_s` is **0, which is uncalibrated**.
2. **Tracker to TCP.** `T_tracker_tcp` is **assumed, not measured**. The assumption is a pure rotation with zero offset (T265 facing along the jaws, upright):
   ```
   R_TRACKER_TCP_ASSUMED = [[0,-1,0],[-1,0,0],[0,0,-1]]
   ```
   The same assumption is what lets the TCP be treated as YAM `tool0` at deployment.
3. **Episode frame.** All poses are expressed relative to the TCP pose at the episode's first frame.
4. **Jaw width.**
   - The separation of ArUco markers 13 and 14 (DICT_4X4_50) in pixels is converted to millimetres with a linear fit from `data/calibration/width-2026-10-02.json`.
   - Fit: gain 0.160 mm/px, offset −4.35 mm, RMS 0.40 mm, max residual 0.62 mm, 5 openings. The 48 mm opening was excluded as non-monotonic.
   - Frames where the markers aren't detected are linearly interpolated, and the ends are held.
   - Exported widths range from **15 to 58 mm**.
   - In the worst episode only **18 %** of frames had a measured width. The rest are interpolated.

### Dataset features

| Key | Shape | Meaning |
|---|---|---|
| `observation.images.wrist` | 480×640×3 | D405 RGB, stored as AV1 video |
| `observation.state` | 11 | Relative proprioception (§4) |
| `observation.ee_pose` | 10 | Current TCP in the episode frame: xyz (m), rot6d, width (m) |
| `observation.gripper_width_measured` | 1 | 1 if markers were detected on this frame, 0 if the width was interpolated |
| `action` | 10 | **Next** TCP pose in the episode frame plus width. Training converts this to a relative chunk. |

`rot6d` is the first two columns of the rotation matrix, flattened column by column:
`r00 r10 r20 r01 r11 r21`. Decoding uses Gram–Schmidt (`rot6d_to_matrix`).

---

## 3. Training

The model was trained on a remote GPU with `train_act_relative.py`. The script's default
paths point to `/home/sra/vrushtee/yumi_act/`. A copy ships in the checkpoint folder.

```bash
python train_act_relative.py \
    --root <local LeRobot copy of vruga/yumi-umi-block-place> \
    --repo-id vruga/yumi-umi-block-place \
    --out outputs/act_relative \
    --chunk 50 --steps 20000 --batch 32 --lr 1e-4 --workers 8 --save-every 5000
```

### What the script does

- **Action windows.** It loads 50 future actions per sample with `delta_timestamps = {action: [0, 1/30, …, 49/30]}`.
- **Relative actions (PD2.1).** It converts each window into poses relative to the current `observation.ee_pose`, inside the training loop and before normalization:
  `rel_k = inv(T_now) · T_k`. Position and rot6d are relative; the width stays absolute (m).
- **Action normalization stats.** `relative_action_stats` computes them over the relative chunks of the whole dataset. The dataset's own action stats are absolute and would be wrong here. Image and state stats come from the dataset.
- **Optimization.** bf16 autocast, AdamW, backbone learning rate 1e-5, gradient clipping at 10.

### Model

The model is LeRobot ACT with its defaults unless listed here:

| Setting | Value |
|---|---|
| Inputs | `observation.images.wrist` (3×480×640), `observation.state` (11) |
| Output | `action` 10-D, `chunk_size = n_action_steps = 50` (1.67 s at 30 fps) |
| Backbone | ResNet18, ImageNet weights |
| Transformer | d_model 512, 8 heads, 4 encoder layers, 1 decoder layer, FFN 3200 |
| VAE | on, latent 32, KL weight 10 |
| Normalization | MEAN_STD for images, state and action |
| Temporal ensembling | off |

**No episodes were held out.** All 22 episodes were used for training.

### Offline check

```bash
cd ~/karma && uv run python ../yumi/scripts/infer_act_relative.py offline \
    --checkpoint ../yumi/data/policies/yumi-umi-block-place-act-relative \
    --dataset ../yumi/data/lerobot/yumi-umi-block-place \
    --episodes 21 --stride 30 --output ../yumi/artifacts/act-relative-offline.json
```

Results on 9 frames of episode 21 (`artifacts/act-relative-offline.json`), measured against
the recorded relative chunks:

| Metric | Mean |
|---|---|
| Position error over the chunk | 7.0 mm |
| Position error at the chunk end (1.67 s) | 9.6 mm |
| Rotation error | 1.12° |
| Width error | 0.41 mm |
| Inference time, CPU | 81 ms |

Episode 21 was in the training set, so this shows the model reproduces its training data.
**It does not measure generalization.**

---

## 4. Model inputs

### Camera: `observation.images.wrist`

- **Format:** D405 colour stream, `rgb8` at 640×480 and 30 fps. It is passed as uint8 HWC and converted to float CHW in [0, 1] (`Policy.__call__`). LeRobot's preprocessor then applies MEAN_STD normalization with the dataset's image stats.
- **No resizing or cropping:** the network sees the full 640×480 frame, as in training.
- **Capture time:** taken as frame arrival minus `--camera-latency`, which defaults to 32 ms and is not measured. This timestamp is the reference for the proprioception lookup.
- **Startup:** the first 30 frames are discarded to let auto-exposure settle. The camera opens **before** the arm is powered, so a busy or missing camera fails while nothing is energized.

### Proprioception: `observation.state` (11)

| Index | Name | Meaning |
|---|---|---|
| 0–2 | `prev_rel_x/y/z` | Position (m) of the TCP **3 frames (100 ms) ago**, in the current TCP frame |
| 3–8 | `prev_rel_r00…r21` | rot6d of that past TCP orientation, in the current TCP frame |
| 9 | `prev_width` | Jaw width 100 ms ago (m) |
| 10 | `width` | Jaw width now (m) |

The current pose relative to itself is the identity, so it's omitted. The state is a
**100 ms motion history plus the gripper**, with no absolute position. This is what makes
the policy independent of where the handheld world origin or the robot base is.

The training and deployment sources:

| | Training (export) | Deployment (live) |
|---|---|---|
| Pose | T265 pose × `T_tracker_tcp` (assumed) | Forward kinematics of the 6 measured YAM joints → `tool0` (`DecoupledIKSolver.fk`, from `vr_teleop_kit`) |
| Rate | T265 stream | Karma arm state, polled every 4 ms into a 2 s `PoseBuffer` |
| Time alignment | Interpolated to the RGB capture time | Interpolated to `t_obs` and `t_obs − 0.1 s` (lerp for position, slerp for rotation). Up to 50 ms of extrapolation is allowed. |
| Width | ArUco marker separation → mm | Karma's normalized gripper position (1 = open) × 95 mm (2 × `YAM_FINGER_MAX`) |
| Stale-state guard | Confidence ≥ 3 and gap ≤ 25 ms | Karma `StateGuard`. A stale state pauses commands; more than 1 s stale raises an error and parks the arm. |

### Output: action chunk (50 × 10)

The output is 50 future TCP poses at 1/30 s spacing, relative to the TCP pose at
observation time. Each pose has relative xyz (m), relative rot6d, and an **absolute** jaw width (m).

---

## 5. Deployment

### Rig

| Item | Value |
|---|---|
| Arm | Right arm of Karma's `yam_bimanual` rig, run alone (`rig.subset(['right'])`) |
| CAN | `can_right`. The script defaults to `can1`, which doesn't exist on this NUC. |
| Wrist camera | Right-wrist D405, **SDK serial `352122271723`** (ASIC `254623070417`) |
| Inference device | CPU. The NUC has no GPU. `--device cuda` falls back automatically. |

`--serial` takes the **SDK** serial because the script opens the camera with
`rs.config.enable_device()`. Karma's `--camera-serial` takes the **ASIC** serial.

| Camera | SDK serial | ASIC serial |
|---|---|---|
| Right wrist D405 | 352122271723 | 254623070417 |
| Left wrist D405 | 409122274921 | 254623070863 |
| Top D435 | 243622071623 | 348523020354 |
| Handheld UMI D405 (training only) | 352122273221 | — |

**How the right-wrist camera was identified (2026-10-03):** the operator covered the right
gripper's camera while mean brightness was printed for each D405. `352122271723` dropped
to about 9–16, and `409122274921` stayed at about 115. To repeat the test, stop anything
that is using the cameras first:

```bash
cd ~/karma && uv run python -c "
import pyrealsense2 as rs, numpy as np, time
pipes = {}
for d in rs.context().devices:
    if 'D405' in d.get_info(rs.camera_info.name):
        s = d.get_info(rs.camera_info.serial_number)
        c = rs.config(); c.enable_device(s); c.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        p = rs.pipeline(); p.start(c); pipes[s] = p
t = time.time()
while time.time() - t < 20:
    print('  '.join(f'{s}: {np.asanyarray(p.wait_for_frames().get_color_frame().get_data()).mean():5.1f}' for s, p in pipes.items()), flush=True)
    time.sleep(0.5)
"
```

### Commands

All commands run from `~/karma`, because the script needs Karma's environment:
LeRobot 0.6.1, `pyrealsense2`, `vr_teleop_kit` and `openpi_control`.

**1. Sim dry run.** This uses the real camera and the real policy with a virtual arm.
Nothing is energized.

```bash
cd ~/karma
uv run python ../yumi/scripts/infer_act_relative.py live --robot sim \
  --serial 352122271723 \
  --speed 0.5 --steps-per-inference 15 \
  --viser --log ../yumi/artifacts/act-relative-sim-$(date +%Y%m%d-%H%M%S).json
```

**2. Live run on the right arm.**

```bash
cd ~/karma
uv run python ../yumi/scripts/infer_act_relative.py live --robot yam \
  --arm right --interface can_right --serial 352122271723 \
  --speed 0.5 --steps-per-inference 15 --max-seconds 60 \
  --viser --log ../yumi/artifacts/act-relative-live-$(date +%Y%m%d-%H%M%S).json
```

Viser runs at http://localhost:8090. It shows the predicted chunk as orange points and
the measured `tool0` as an axis frame. **Ctrl+C** stops commands, parks the arm at
`home_pos`, and powers it down.

Put a timestamp in `--log`. The file is overwritten on every run, which is why only
the last live rollout from 2026-10-03 survives.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--speed` | 0.5 | Playback rate relative to training. At 0.5, each step lasts 1/15 s, so a chunk plays over 3.3 s instead of 1.67 s. |
| `--steps-per-inference` | 15 | Receding horizon: execute this many steps, then observe again. At speed 0.5 that's about 1 s per cycle. |
| `--camera-latency` | 0.032 | Seconds subtracted from frame arrival to get capture time |
| `--exec-latency` | 0.05 | Command-to-motion delay. Steps due before now + this are skipped (PD1.2). |
| `--max-step-rad` | 0.10 | Karma's per-30 Hz-tick joint clamp, spread over the IK ticks (0.015 rad per tick at 200 Hz) |
| `--max-effector-step` | 0.30 | Karma's per-tick gripper clamp |
| `--max-chunk-reach` | 0.25 | Chunks whose TCP moves more than this many metres are rejected, and the arm holds |
| `--min-z` | −1 | Floor for `tool0` z in the arm base frame. **Effectively off by default.** |
| `--ik-hz`, `--mu`, `--gripper-speed` | 200, 0.005, 2.0 | `EEFollower` and `DecoupledIKSolver` settings |
| `--max-seconds` | 0 | 0 runs until Ctrl+C |
| `--no-park` | off | Power down in place instead of parking at `home_pos` |

### Control loop (one cycle)

1. **Observe.** Read a D405 frame and set `t_obs = arrival − camera_latency`.
2. **Build the state.** Look up `T_now`, `w_now` at `t_obs` and `T_prev`, `w_prev` at `t_obs − 0.1 s` in the pose buffer.
3. **Infer.** Run ACT to get the relative chunk `rel` (50×10). If it contains NaN or Inf, the run stops and the arm parks.
4. **Check reach.** If any step moves the TCP more than `--max-chunk-reach`, reject the chunk and observe again.
5. **Compose.** `targets = T_now @ pose_to_T(rel)` (PD2.1).
6. **Schedule.** `due_k = t_obs + (k+1)/(30·speed)`. Skip every step whose due time is before `now + exec_latency` (PD1.2), then run the next `steps-per-inference` steps.
7. **Execute each step.** At its due time:
   - clamp z to `--min-z`;
   - convert width to gripper closedness, `clip(1 − w / 0.095, 0, 1)`;
   - call `follower.set_target(...)`.
8. **Track at 200 Hz.** `EEFollower` solves IK. Each joint command is clamped by the per-tick step limit and the joint limits, the gripper by its step limit, and nothing is sent unless the state is fresh (≤ 250 ms). Then `arm.command(PositionCommand)` is sent to Karma.

### Shutdown

The lifecycle matches Karma's `inference` command:

- the camera opens before the motors;
- on Ctrl+C or any error, a stop flag is set first, so nothing can race the park;
- the arm then parks at `home_pos` and powers down. That step is registered first, so it always runs last, even if closing the camera, Viser or the follower fails.

---

## 6. Results (2026-10-03)

The operator ran several live rollouts on the right arm and called them decent. No
success rate was recorded. Only the last live run's log survives, because `--log` was overwritten.

| | Sim dry run | Last live run |
|---|---|---|
| Log | `artifacts/act-relative-sim.json` | `artifacts/act-relative-live.json` |
| Duration / chunks | 266 s / 254 | 59 s / 57 (stopped by `--max-seconds 60`) |
| Inference, median / p90 / max | 84 / 98 / 116 ms | 90 / 109 / 118 ms |
| Steps skipped per chunk, median (max) | 2 (2) | 2 (3) |
| Cycle time, median | 1.05 s | 1.05 s |
| Chunk-end displacement, median / p90 | 21 / 32 mm | 56 / 154 mm |
| Max TCP reach in a chunk | 66 mm | 188 mm (under the 250 mm limit) |
| Predicted width range | 43.6–54.3 mm | 42.2–54.5 mm |

What the numbers show:

- **CPU inference is fast enough at speed 0.5.** About 90 ms costs only 2 of 50 steps per chunk, and no step was late enough to cause a gap longer than 1.1 s.
- **Live chunks are much larger than sim chunks.** On the real arm the policy commanded much bigger motions, as the task needs. In sim, only the virtual arm moves: the wrist camera stays still, so the view never changes in response to the policy.
- **The gripper barely closed in the logged run.** Widths stayed at 42–55 mm, against 15–58 mm in training. That maps to gripper closedness of about 0.43–0.56, so this log shows no full grasp. See the gripper-scale risk below.

To summarize a log:

```bash
cd ~/karma && uv run python -c "
import json, sys, numpy as np
L = json.load(open(sys.argv[1]))
inf = np.array([x['infer_s'] for x in L])*1000; rel = np.array([x['rel'] for x in L])
print(len(L), 'chunks; infer ms med %.0f; skip med %d; reach max %.0f mm; width %.1f-%.1f mm' % (
  np.median(inf), np.median([x['skipped'] for x in L]),
  np.linalg.norm(rel[:, :, :3], axis=-1).max()*1000, rel[:, :, 9].min()*1000, rel[:, :, 9].max()*1000))
" ../yumi/artifacts/act-relative-live.json
```

---

## 7. Known risks and gaps

1. **No table floor.** `--min-z` defaults to −1 m. Measure the table height in the right arm's base frame and pass something like `--min-z <table + 0.02>`.
2. **Assumed tracker-to-TCP transform.** `R_TRACKER_TCP_ASSUMED` sets both the training action frame and its mapping onto `tool0`. If it's wrong, the commanded motion is rotated by the error. It has not been measured. The rollouts working suggests it is roughly right, but that isn't proof.
3. **Different camera at deployment.** Training used the handheld D405 (`352122273221`); deployment uses the right-wrist D405 (`352122271723`). Mounting pose, field of view of the fingers, and exposure all differ. The right-wrist camera read noticeably dark: mean brightness about 34 uncovered, against about 115 for the left wrist. Check its Viser view before each session.
4. **Gripper scale mismatch.**
   - The live width is the robot's normalized gripper position × 95 mm. Training widths only go up to 58 mm.
   - A fully open YAM therefore reports 95 mm in `observation.state`, outside the training distribution.
   - A predicted 58 mm "open" commands the gripper only about 39 % closed, and the YAM never opens fully.
   - The YAM finger opening hasn't been measured against the UMI jaw opening.
5. **Uncalibrated timing.** `camera_time_offset_s = 0` in training, and `--camera-latency 0.032` / `--exec-latency 0.05` at deployment, are all estimates.
6. **Interpolated width labels.** In the sparsest episode only 18 % of width labels were measured.
7. **No held-out evaluation.** The offline numbers are on training data.
8. **Single task, small dataset.** 22 demonstrations of 3.9 minutes in total, one block and one board.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `xioctl(VIDIOC_S_FMT) failed, errno=16 … Device or resource busy` | Another process has the D405 open (often a previous sim/live run) | Find it with `for v in /dev/video*; do fuser $v; done`, then stop it |
| RealSense error on start when `--serial` is omitted | `--serial` default `352122273221` is the handheld UMI camera, which isn't connected | Pass `--serial 352122271723` |
| Power-up fails on `can1` | The script default doesn't exist on this NUC | Pass `--interface can_right` and check with `ip -br link` |
| Karma preflight: `serial … not on the bus` | An SDK serial was given to Karma's `--camera-serial` | Karma needs the **ASIC** serial (see the table in §5) |
| `chunk rejected: reaches … mm` repeatedly | The policy wants a large jump, often from an out-of-distribution view or start pose | Reset the arm and scene to look like a training start, and check the camera view |
| `pose buffer does not cover t=…` | No fresh arm state around the frame time | Check CAN and Karma state. The run stops and parks. |
| `CUDA unavailable; running on CPU` | NUC has no GPU | Expected. CPU inference is about 90 ms. |
