"""LeRobot v3 writer for actual right-arm replay observations, using Karma's sink."""
import json
import time
from pathlib import Path

import numpy as np


class ReplayDataset:
    def __init__(self, root, repo_id, readers, *, fps=30, task='Replay accepted UMI demonstration', provenance=None):
        from openpi_control.record import LeRobotSink, arm_feature_names, build_features, camera_shapes
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(f'Dataset already exists: {self.root}')
        self.readers, self.fps, self.task = readers, fps, task
        extra = {
            'observation.elapsed_s': {'dtype': 'float32', 'shape': (1,), 'names': ['elapsed_s']},
            'observation.state_age_s': {'dtype': 'float32', 'shape': (1,), 'names': ['state_age_s']},
            'replay_sample_index': {'dtype': 'int64', 'shape': (1,), 'names': ['sample_index']},
        }
        features = build_features(arm_feature_names(['right'], {'right': 6}), camera_shapes(readers), extra=extra)
        self.sink = LeRobotSink(repo_id=repo_id, fps=fps, features=features, robot_type='Yam', root=self.root)
        self.started = self.next_due = None
        self.frames = 0
        self.last_camera_count = None
        self.last_camera_change = None
        self.closed = False
        self.provenance = provenance or {}

    def add(self, state, command, sample_index):
        now = time.monotonic()
        if self.started is None:
            self.started = self.next_due = self.last_camera_change = now
        if now < self.next_due:
            return
        if not state.is_fresh(.1):
            raise RuntimeError('Dataset joint feedback is stale; stopping replay')
        reader = self.readers['right_wrist']
        image = reader.latest()
        count = reader.frames_read
        if image is None or count == self.last_camera_count:
            if now-self.last_camera_change > .5:
                raise RuntimeError('D405 stopped delivering new images; stopping replay')
            return
        if state.effector is None:
            raise RuntimeError('Dataset needs measured gripper feedback')
        observation = np.r_[state.joints.position_rad, state.effector.position].astype(np.float32)
        action = np.asarray(command, dtype=np.float32)
        if observation.shape != (7,) or action.shape != (7,) or not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise ValueError('Non-finite or invalid measured state/action')
        if reader.pixel_format == 'bgr8':
            image = np.ascontiguousarray(image[:, :, ::-1])
        elif reader.pixel_format != 'rgb8':
            raise ValueError(f'Unsupported camera pixel format: {reader.pixel_format}')
        self.sink.add_frame({
            'observation.state': observation,
            'action': action,
            'observation.images.right_wrist': image,
            'observation.elapsed_s': np.array([now-self.started], dtype=np.float32),
            'observation.state_age_s': np.array([state.age_s], dtype=np.float32),
            'replay_sample_index': np.array([sample_index], dtype=np.int64),
            'task': self.task,
        })
        self.frames += 1
        self.last_camera_count, self.last_camera_change = count, now
        self.next_due += 1/self.fps
        if self.next_due < now:
            self.next_due = now+1/self.fps  # no fabricated catch-up frames

    def close(self, completed=False):
        if self.closed:
            return
        self.closed = True
        try:
            if self.frames:
                self.sink.save_episode()
            else:
                self.sink.discard_episode()
        finally:
            self.sink.finalize()
        info = json.loads((self.root/'meta/info.json').read_text())
        if not str(info.get('codebase_version', '')).startswith('v3'):
            raise RuntimeError('Writer did not create a LeRobot v3 dataset')
        provenance = dict(self.provenance, frames=self.frames, replay_completed=completed,
            nominal_fps=self.fps, capture='D405 RGB only; no depth stream recorded',
            synchronization='Latest camera frame with fresh measured robot state at record tick; no exposure-time registration. Actual sampling elapsed time stored per frame.',
            state='Six measured right-arm joints (radians) plus measured normalized gripper opening',
            action='Actual bounded command: six joint targets plus normalized gripper opening',
            scope='Replay portion only; excludes approach and native parking')
        (self.root/'replay-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        print(f'LeRobot v3 saved: {self.root} ({self.frames} frames; completed={completed})', flush=True)
