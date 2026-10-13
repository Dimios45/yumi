Collect ten accepted handheld demonstrations from a home reference:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_10_demonstrations.py \
  --config configs/umi.collection-provisional.json \
  --profile configs/yam.collection-fingertip.json \
  --task "Pick up the green block and place it on the wooden board" \
  --count 10 \
  --output data/collection-10-new --allow-provisional
```

The configuration above is explicitly provisional: its TCP geometry, camera timing,
gripper fit and capture-to-robot registration still need physical validation. Replace
it with a measured configuration when available. `--preflight` checks configuration
and dependencies without opening sensors or creating a collection.

Press SPACE to start each attempt and SPACE again to stop it; q quits (during a
recording it also stops and processes that attempt). `--seconds N` adds an
auto-stop. Return the handheld to the same starting pose first, and keep the gripper markers visible throughout. The YAM home
reference is six zero joint angles. This script does not connect to motors or move
the robot. Full seven-component labels are six offline IK arm angles plus physical
jaw aperture in metres; they are not measured robot feedback.

Sensors warm up for 20 seconds, and recording requires
fresh confidence 3/3 and valid width for one second. The browser preview uses port
8080. The supplied research profile uses AprilTag corner refinement and the existing
five-point October 2 width fit, covering approximately 11–47 mm only. It does not
calibrate fully closed jaws. Replace it with a fresh measured calibration for tasks
outside this range; out-of-range readings are rejected.

Captures pass strict timestamp, pose continuity, confidence, aperture, depth timing,
full-pose IK, model collision and joint rate checks. Rejected attempts remain under
`raw/`, with rejection reasons. Up to 30 attempts are allowed to obtain ten accepted
episodes; quitting early retains raw data and a report but does not publish a
complete paired export. At most 0.2 seconds may be trimmed at either boundary;
internal gaps or large missing portions cause rejection. Inspect the saved source
frame mappings and rejection reports before training.

Outputs are `datasets/ee` (10-component XYZ, rotation columns and aperture state
and action) and `datasets/joint7` (six arm radians and aperture). Both use the
official LeRobot v3 writer, include wrist video, episode-relative and T265-world
TCP pose, robot-retargeted TCP pose, relative actions, sensor timestamps, IK errors,
and aperture measurement flags. Actions are the next captured state. Original
sensor logs and native uint16 depth are retained under dataset `extras/`. Absolute
T265-world pose does not establish absolute robot-base registration.

Each dataset is loaded again through LeRobot, every frame is decoded, and action
continuity and matching episode/frame counts are checked before staging directories
are renamed. Existing datasets are never overwritten. If the host cache is not
writable, set `HF_HOME` to a writable directory before running.

Physical tests still required: measured TCP and tool axes, independent width
validation over the intended aperture range, RGB-to-pose latency, tracking accuracy
against an external reference, workspace/table geometry, and measured robot homing
and feedback. Confidence 3 alone does not establish tracking accuracy.
