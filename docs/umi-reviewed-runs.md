# UMI runs with a fixed task start

Run commands from `~/yumi`. The profile `configs/yam.umi-start.json` stores the
successful corrected replay's first task posture (radians):
`[-0.00000436, 0.681154, 0.564064, 0.250521, -0.00000657, 0.027388]`.
This is the task start after the approach, separate from zero joints and motor
shutdown parking. The existing Karma model and `yam_motion.py` smooth planner
and streamer are reused. No motor calibration/home files are changed.

## One interactive command

```bash
cd ~/yumi
.venv/bin/python scripts/collect_umi_run.py
```

`bash scripts/record_umi_run.sh` also launches this same workflow.
Open `http://192.168.0.196:8080` and keep the terminal open.

1. Warm up tracking to confidence 3. Hold the handheld in the demonstrated
   starting orientation. SPACE starts recording only at confidence 3; SPACE stops it.
2. Capture closes automatically and the existing smooth replay starts on the
   same Viser port. Refresh the browser when the replay is ready. Compare the
   camera, requested TCP and simulated achieved motion.
3. Use the buttons at the top of Viser:
   - **Accept & next** saves an accepted manifest and opens the next recording.
   - **Accept & finish** saves an accepted manifest and exits.
   - **Retry** retains this attempt as pending and opens another recording.
   - **Finish** retains this attempt as pending and exits.

Each next recording reopens the cameras; refresh Viser and wait for confidence 3 before SPACE.
Nothing is accepted automatically. During capture, q exits (an active recording
is finalized and reviewed). Ctrl+C stops the workflow and retains raw attempts.
Low-confidence tracking still opens a diagnostic replay with acceptance disabled.
If capture or review cannot run, the terminal offers retry or finish.

Raw files are retained under `data/umi-runs/pending` so review is possible.
Acceptance creates a manifest in `data/umi-runs/accepted` referencing the raw
capture and its report. No files are deleted or moved and no training export
runs automatically. Accepted buttons are disabled unless >=95% of displayed
recorded poses are within 10 mm/5 degrees, maxima are <=25 mm/10 degrees,
>=95% of jaw measurements are valid, tracking is valid throughout, and no floor
correction was needed. These are workflow thresholds, not proof of physical
accuracy. Changed raw data, profile or replay code requires another review.

To resume review of a retained episode with the same interactive workflow:

```bash
.venv/bin/python scripts/collect_umi_run.py --review data/umi-runs/pending/RUN/episode-000
```

If port 8080 is occupied, stop its viewer or add `--port 8083` to the command.
No CAN connection or robot command is part of this collection/review workflow.

## Existing inference entry point

For a future operator-run policy command, add
`--settings configs/yam.umi-run.json` to the existing
`scripts/infer_act_relative.py live` command. It loads the new task start through
`--umi-start-profile` and uses the existing smooth approach before policy
execution. Explicit `--umi-start-profile configs/yam.umi-start.json` also works.
Simulation initializes at these joints; hardware starts from measured joints
and plans its approach. Demo/session replay has its own recorded-path approach.
No physical robot command was run to install or test this profile.

Gravity alignment preserves the measured handheld tilt and sets its initial
heading to robot +X, with the same workspace anchor as the corrected replay.
Changing the handheld starting tilt therefore changes the first requested pose;
the saved robot task posture is fixed and the planner approaches that request.

TCP registration and jaw calibration remain provisional. Width coverage ends
at the measured 58 mm; the asserted 74 mm maximum has no paired calibration.
Invalid widths are held only for visualization and counted as invalid in the
acceptance gate. An accepted trajectory is not a physical calibration approval.
