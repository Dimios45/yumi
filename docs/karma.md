# Karma: how yumi runs on the YAM arm

[Karma](https://github.com/SRA-VJTI/karma) is SRA-VJTI's robot control stack
for bimanual YAM and SO100/SO101 arms. It covers Quest VR teleoperation,
MolmoAct2 inference, rollout recording and HITL data collection.
yumi does not reimplement any of that. Its robot-side scripts run **inside
Karma's environment** and use Karma for the hardware, IK and safety layers.

```
~/yumi   capture, calibration, export, training scripts, policy inference
~/karma  YAM motor runtime, CAN, rigs, IK, safety, LeRobot 0.6.1, viser, pyrealsense2
```

Every robot or LeRobot command in yumi runs like this:

```bash
cd ~/karma && uv run python ../yumi/scripts/<script>.py ...
```

That makes Karma's uv environment the interpreter, with LeRobot 0.6.1, a current
`pyrealsense2`, MuJoCo, viser, `openpi_control` and `vr_teleop_kit`, while the
script comes from yumi. No `PYTHONPATH` setup and no install of yumi into Karma
is needed.

The handheld capture (`uv run yumi ...` in `~/yumi`) does **not** use Karma. It
needs the T265, which only works with the pinned `pyrealsense2==2.53.1` in yumi's
own Python 3.10 environment.

## What yumi uses from Karma

| Karma module | Used for | yumi scripts |
|---|---|---|
| `openpi_control.rigs.resolve_rig` | The `yam_bimanual` rig, cut down to one arm with `.subset(['right'])` and `.with_interfaces({'right': 'can_right'})` | inference, gripper calibration, replay |
| `openpi_control.cli.power_up` / `power_down` / `settle_arm_states` | Energize the arm, read a settled state, park at `home_pos` and de-energize on exit | `infer_act_relative.py`, `measure_yam_gripper.py`, `replay_accepted_yam.py` |
| `openpi_control.types.PositionCommand` | Joint position + gripper command sent to the arm | same |
| `openpi_control.safety` | `StateGuard` (no command on stale state; error and park after 1 s), `joint_limits`, `MAX_STATE_AGE_S`, gripper start report | same |
| `vr_teleop_kit.ik.DecoupledIKSolver` | Forward kinematics of `tool0` for `observation.state`; IK for the `--ik follower` path | inference, replay, `yam_motion.py` |
| `vr_teleop_kit.ik.EEFollower` | 200 Hz end-effector follower with Karma's per-tick joint and gripper clamps | inference (`--ik follower`), `replay_yam.py`, `filter_training_ik.py` |
| `openpi_control.viz` | YAM arm scene and camera panels in Viser | `yam_motion.py`, `replay_accepted_yam.py` |
| `openpi_control.record`, `openpi_control.inference` | LeRobot sink and bounded chunk executor | `yam_replay_dataset.py`, `replay_accepted_yam.py` |
| LeRobot 0.6.1 (Karma's lockfile) | Reading and writing LeRobot v3 datasets, ACT policy classes | `export_umi_lerobot.py`, `infer_act_relative.py`, `make_rollout_video.py` |

yumi's own `scripts/yam_motion.py` provides the default `--ik smooth` path:
converged 6-DoF pose IK, lift arcs, the elbow-assist tilt, and rate-limited joint
streaming. It still sends every command through Karma's `PositionCommand`, after
Karma's limits.

The lifecycle matches Karma's own `karma inference`:

- the camera opens before the motors, so a missing camera fails with nothing energized;
- a stop flag is set before anything else on Ctrl+C or an error;
- the park and power-down are registered first, so they run last, even if closing
  the camera or Viser fails.

## Setting Karma up for yumi

Follow Karma's [README](https://github.com/SRA-VJTI/karma#readme) and
[YAM setup](https://github.com/SRA-VJTI/karma/blob/main/docs/yam-setup.md). The parts yumi needs:

```bash
# 1. Clone next to yumi; yumi's commands use the relative path ../yumi
git clone https://github.com/SRA-VJTI/karma ~/karma
cd ~/karma

# 2. Build deps and sync (Ubuntu, Python 3.12/3.13, USB 3 for RealSense)
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
UV_HTTP_TIMEOUT=180 uv sync --locked
sudo usermod -aG dialout,video "$USER"     # log out and back in

# 3. YAM model and meshes (i2rt checkout inside karma, git-ignored)
git clone https://github.com/i2rt-robotics/i2rt.git
uv run karma-viz --fetch-meshes --model Yam
```

**CAN buses.** Each arm has its own USB-CAN adapter. `can0`/`can1` follow USB
enumeration order, so this rig pins them to `can_left` and `can_right` with a
udev rule keyed on the adapter serial (Karma `docs/yam-setup.md`). Bring them up
at 1 Mbit/s after every boot and check the right arm:

```bash
sudo ip link set can_right up type can bitrate 1000000
sudo ip link set can_left  up type can bitrate 1000000
ip -br link show type can
uv run karma doctor --rig yam_bimanual --interface-override right=can_right --probe
```

yumi's scripts take `--arm right --interface can_right`. Both are the defaults
inherited from `configs/yam.replay.json`.

**Cameras.** List them from Karma's environment:

```bash
uv run python -c 'import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])'
```

yumi uses the **SDK serials** printed here: right wrist `352122271723`, top
`243622071623`. Karma's own `--camera-serial` flags take the **ASIC** serials
(`254623070417`, `348523020354`). Only one process can open a RealSense camera,
so stop `karma teleop`/`inference` before running a yumi script.

**Extra packages per command.** The ACT + CLIP policy needs `transformers`. Add
it per command instead of changing Karma's lockfile:

```bash
uv run --with transformers --with huggingface_hub==1.28.0 python ../yumi/scripts/infer_act_relative.py ...
```

**Zeros and parking.** YAM uses firmware motor zeros. `home_pos` from Karma's
packaged instance config is where yumi parks on exit. yumi never writes zeros
or calibration files.

## Pinned Karma for offline planning

`uv run yumi plan-robot` (offline IK feasibility in yumi's own environment)
loads the YAM model from a pinned Karma and i2rt checkout under `research/`
instead of `~/karma`:

```bash
git clone https://github.com/sra-vjti/karma research/karma
git -C research/karma checkout b4f06f6d645755e605b6c0aec7c10af3d2c911d6
git clone https://github.com/i2rt-robotics/i2rt research/i2rt
git -C research/i2rt checkout 5d47b358bafb30c65e397f2ece506550a0db4594
uv sync --locked --extra robot --extra dev
```

See [single-arm-umi.md §4](single-arm-umi.md#4-offline-robot-feasibility).
