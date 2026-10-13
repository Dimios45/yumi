"""Gate synchronized session labels using the corrected replay's actual IK path."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import replay_yam as replay

from yumi.geometry import transform
from yumi.umi import make_chunks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    summaries = []
    for path in sorted(a.prepared.glob('episode_*.json')):
        doc = json.loads(path.read_text())
        previous_report = json.loads((a.replay/path.name).read_text())
        anchor = replay.anchor_transform(previous_report['anchor_xyz_m_yaw_deg'])
        solver = replay.DecoupledIKSolver(mu=.005, max_dq_per_joint=replay.MAX_DQ)
        follower = replay.EEFollower(solver, freq=200, q_init=replay.READY, gripper_max_speed=3.)
        frames = doc['frames']
        for f in frames:
            f['valid_before_ik'] = f['valid']
            if 'T_world_tcp' in f:
                f['T_robot_target'] = (anchor@transform(f['T_world_tcp'])).tolist()
        first = next(f for f in frames if 'T_robot_target' in f)

        def command(f, follower=follower):
            target = np.asarray(f['T_robot_target'])
            width = f.get('opening_m')
            closed = None if width is None else float(np.clip(1-width/(2*replay.YAM_FINGER_MAX),0,1))
            follower.set_target(target[:3,3], replay.Rotation.from_matrix(target[:3,:3]).as_quat(), closed)

        command(first)
        for _ in range(600):
            follower.step()
        prior = None
        for f in frames:
            if 'T_robot_target' in f:
                command(f)
            dt = 1/30 if prior is None else f['time_s']-prior['time_s']
            for _ in range(max(1, round(dt*200))):
                follower.step()
            position, quat = solver.fk(follower.qpos)
            actual = replay.homogeneous(replay.Rotation.from_quat(quat[[1,2,3,0]]).as_matrix(),position)
            f['joint_positions_rad'] = follower.qpos.tolist()
            f['T_robot_actual'] = actual.tolist()
            if 'T_robot_target' in f:
                pe, re = replay.pose_error(np.asarray(f['T_robot_target']),actual)
                f['ik_error'] = [pe/1000,np.radians(re)]
                if pe>replay.POS_OK_MM or re>replay.ROT_OK_DEG:
                    f['valid'] = False
                    f['reasons'].append('IK tracking exceeds 10mm / 5deg')
            if prior is not None:
                q = np.asarray(f['joint_positions_rad'])
                prev_q = np.asarray(prior['joint_positions_rad'])
                if dt<=0 or np.max(np.abs(q-prev_q))/dt>3.:
                    for row in (prior,f):
                        row['valid'] = False
                        row['reasons'].append('Joint speed exceeds research bound 3rad/s')
            prior = f
        chunks = make_chunks(frames,doc['history'],doc['horizon'],doc['fps'])
        # Fixed-rate training requires the entire horizon to match the expected times.
        nominal = np.arange(1,doc['horizon']+1)/doc['fps']
        doc['samples'] = [c for c in chunks if np.max(np.abs(np.asarray(c['action_time_offsets_s'])-nominal))<=.012]
        doc['ik_filter'] = {'position_threshold_m':.01,'rotation_threshold_rad':float(np.radians(5)),
                            'max_joint_speed_rad_s':3.,'solver':'Karma DecoupledIKSolver + EEFollower, 200Hz',
                            'anchor_xyz_m_yaw_deg':previous_report['anchor_xyz_m_yaw_deg'],
                            'collision_checked':False,'registration_verified':False}
        errors = Counter(reason for f in frames for reason in set(f['reasons']))
        summary = {'episode':doc['source_episode'],'frames':len(frames),
                   'valid_before_ik':sum(f['valid_before_ik'] for f in frames),
                   'valid_after_ik':sum(f['valid'] for f in frames),'chunks':len(doc['samples']),
                   'invalid_reasons':dict(errors)}
        summaries.append(summary)
        (a.output/path.name).write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
        print(json.dumps(summary),flush=True)
    result = {'source_prepared':str(a.prepared.resolve()),'episodes':summaries,
              'research_training_ready':any(e['chunks'] for e in summaries),
              'calibration_verified':False,'timing_verified':False,'hardware_validated':False,
              'prepared_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.prepared.glob('episode_*.json')}}
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
