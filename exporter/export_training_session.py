"""Export strict, continuous session segments for the project's relative ACT workflow."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
from export_demonstrations import encode, matrix
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset


def valid_runs(doc):
    """Keep only spans which contain a full valid history and action horizon."""
    frames=doc['frames']
    runs=[]
    start=None
    valid_anchors={s['anchor_frame'] for s in doc['samples']}
    for i in range(len(frames)+1):
        if i<len(frames) and frames[i]['valid']:
            if start is None:
                start=i
        elif start is not None:
            anchors=[j for j in range(start+doc['history']-1,i-doc['horizon']) if j in valid_anchors]
            if anchors:
                runs.append((frames[start:i],anchors))
            start=None
    return runs


def export(prepared, root, repo_id):
    if CODEBASE_VERSION!='v3.0':
        raise ValueError('Official LeRobot v3 required')
    stage=root.with_name(root.name+'.partial')
    if root.exists() or stage.exists():
        raise FileExistsError('Use a fresh export directory')
    docs=[json.loads(p.read_text()) for p in sorted(prepared.glob('episode_*.json'))]
    first=docs[0]
    horizon=first['horizon']
    if any(d['fps']!=30 or d['horizon']!=horizon for d in docs):
        raise ValueError('Inconsistent input contract')
    ee={'dtype':'float32','shape':(10,),'names':['x_m','y_m','z_m','r00','r10','r20','r01','r11','r21','aperture_m']}
    features={
        'observation.images.wrist':{'dtype':'video','shape':(480,640,3),'names':['height','width','channels']},
        'observation.state':{'dtype':'float32','shape':(11,),'names':None},
        'observation.ee_pose':ee,'action':ee,'action.ee_relative':ee,
        'observation.ee_pose.world':ee,'observation.ee_pose.robot':ee,
        'observation.umi_width_m':{'dtype':'float32','shape':(1,),'names':['aperture_m']},
        'observation.gripper_open':{'dtype':'float32','shape':(1,),'names':['native_open']},
        'observation.gripper_width_measured':{'dtype':'float32','shape':(1,),'names':['measured']},
        'observation.joints_ik':{'dtype':'float32','shape':(6,),'names':[f'joint{i}_rad' for i in range(1,7)]},
        'observation.ik_error':{'dtype':'float32','shape':(2,),'names':['position_m','rotation_rad']},
        'observation.sensor_time':{'dtype':'float64','shape':(2,),'names':['capture_device_s','capture_corrected_s']},
        'action.time_offset':{'dtype':'float32','shape':(1,),'names':['seconds']},
        'observation.source_frame':{'dtype':'int64','shape':(1,),'names':['frame']},
        'observation.source_episode':{'dtype':'int64','shape':(1,),'names':['episode']},
    }
    ds=LeRobotDataset.create(repo_id=repo_id,root=stage,fps=30,features=features,
        robot_type='umi_yam_relative_act_research',video_backend='pyav',vcodec='h264',image_writer_threads=2)
    mappings=[]
    indices=[]
    global_row=0
    caps={}
    last={}
    try:
        for doc in docs:
            for fs,anchors in valid_runs(doc):
                source=doc['source_episode']
                mapping={'output_episode':len(mappings),'source_episode':source,'start_frame':fs[0]['frame_index'],
                         'end_frame':fs[-1]['frame_index'],'rows':len(fs)-1,'first_row':global_row}
                mappings.append(mapping)
                origin=np.linalg.inv(np.asarray(fs[0]['T_world_tcp']))
                poses=[origin@np.asarray(f['T_world_tcp']) for f in fs]
                for i in range(len(fs)-1):
                    f,nxt=fs[i],fs[i+1]
                    prev=max(0,i-3)
                    delta=np.linalg.inv(poses[i])@poses[i+1]
                    past=np.linalg.inv(poses[i])@poses[prev]
                    path=str(Path(doc['source_root'])/f['video_path'])
                    if path not in caps:
                        caps[path]=cv2.VideoCapture(path)
                    cap=caps[path]
                    index=f['video_frame_index']
                    if last.get(path,-2)+1!=index:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                    ok,bgr=cap.read()
                    if not ok:
                        raise ValueError(f'Cannot read source image {path}:{index}')
                    last[path]=index
                    ds.add_frame({
                        'observation.images.wrist':cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),
                        'observation.state':np.r_[encode(past,fs[prev]['opening_m']),f['opening_m']].astype(np.float32),
                        'observation.ee_pose':encode(poses[i],f['opening_m']),
                        'action':encode(poses[i+1],nxt['opening_m']),
                        'action.ee_relative':encode(delta,nxt['opening_m']),
                        'observation.ee_pose.world':encode(np.asarray(f['T_world_tcp']),f['opening_m']),
                        'observation.ee_pose.robot':encode(np.asarray(f['T_robot_target']),f['opening_m']),
                        'observation.umi_width_m':np.array([f['opening_m']],np.float32),
                        'observation.gripper_open':np.array([f['opening_m']/.053],np.float32),
                        'observation.gripper_width_measured':np.ones(1,np.float32),
                        'observation.joints_ik':np.array(f['joint_positions_rad'],np.float32),
                        'observation.ik_error':np.array(f['ik_error'],np.float32),
                        'observation.sensor_time':np.array([f['device_s'],f['time_s']],np.float64),
                        'action.time_offset':np.array([nxt['time_s']-f['time_s']],np.float32),
                        'observation.source_frame':np.array([f['frame_index']],np.int64),
                        'observation.source_episode':np.array([int(source.split('_')[-1])],np.int64),
                        'task':doc['task']})
                    # ACT state uses 3-frame history, so require all of it plus future actions.
                    if f['frame_index'] in anchors and i>=3 and i+horizon<=len(fs)-1:
                        indices.append((global_row,source))
                    global_row+=1
                ds.save_episode()
                print(source,mapping['start_frame'],mapping['end_frame'],mapping['rows'],flush=True)
        if not mappings or not indices:
            raise ValueError('No complete training windows')
        sources=sorted({source for _,source in indices})
        if len(sources)<2:
            raise ValueError('Need separate source episodes for train and validation')
        held_out=sources[-1]
        split={'horizon':horizon,'history_step_frames':3,'held_out_source_episodes':[held_out],
               'train_indices':[i for i,s in indices if s!=held_out],
               'validation_indices':[i for i,s in indices if s==held_out]}
        (stage/'training-split.json').write_text(json.dumps(split,indent=2)+'\n')
        contract={'contract':'yumi.act.relative_session.v1','source_mapping':mappings,
            'fps':30,'horizon':horizon,'observation_state':'11D: previous TCP relative to now (3 frames / ~100ms) and past/current UMI aperture metres',
            'action_semantics':'next absolute pose in segment-initial corrected YAM tool axes; trainer converts entire future chunk relative to current EE',
            'gripper_semantics':'measured UMI aperture metres; native_open=aperture/0.053 is retained separately; deployment must use this mapping rather than treat UMI metres as robot physical aperture',
            'filters':'confident bracketed timestamp-aligned pose, image gaps, raw pose/width speed, measured in-range aperture, <=10mm/5deg IK; fixed-horizon timing within12ms',
            'registration_verified':False,'calibration_verified':False,'timing_verified':False,'collision_checked':False,'hardware_validated':False,
            'research_training_ready':True,'training_indices_exclude_history_and_future_padding':True,
            'calibration':first['calibration'],'source_manifest_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared.glob('episode_*.json')}}
        (stage/'meta/umi_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
        shutil.copyfile(prepared/'audit.json',stage/'quality-audit.json')
    finally:
        ds.finalize()
        if ds.image_writer:
            ds.image_writer.stop()
        for cap in caps.values():
            cap.release()
    report=verify(stage,repo_id)
    (stage/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    stage.rename(root)
    return report


def verify(root,repo_id):
    ds=LeRobotDataset(repo_id,root=root,video_backend='pyav')
    previous=None
    for i in range(len(ds)):
        row=ds[i]
        ep=int(row['episode_index'])
        pose=row['observation.ee_pose'].numpy()
        action=row['action'].numpy()
        matrix(pose);matrix(action)
        if previous and previous[0]==ep:
            assert np.allclose(previous[1],pose,atol=1e-6)
            assert int(row['observation.source_frame'].item())==previous[2]+1
        assert row['observation.state'].shape==(11,) and row['action'].shape==(10,)
        assert row['observation.images.wrist'].shape==(3,480,640)
        assert row['observation.images.wrist'].isfinite().all()
        assert np.allclose(matrix(row['action.ee_relative'].numpy()),np.linalg.inv(matrix(pose))@matrix(action),atol=1e-5)
        assert float(row['observation.gripper_width_measured'].item())==1
        assert float(row['observation.ik_error'][0])<=.010001
        assert float(row['observation.ik_error'][1])<=np.radians(5)+1e-6
        previous=ep,action,int(row['observation.source_frame'].item())
    count=0
    for p in (root/'videos/observation.images.wrist').rglob('*.mp4'):
        with av.open(str(p)) as video:
            count+=sum(1 for _ in video.decode(video=0))
    assert count==len(ds)
    split=json.loads((root/'training-split.json').read_text())
    train=set(split['train_indices']);val=set(split['validation_indices']);assert train and val and not train&val
    for i in sorted(train|val):
        anchor=ds[i]
        assert i+split['horizon']-1<len(ds)
        future=ds[i+split['horizon']-1]
        assert int(anchor['episode_index'])==int(future['episode_index'])
        a=np.asarray(ds.hf_dataset[i]['observation.sensor_time'])
        nominal=np.arange(split['horizon'])/30
        times=np.array([np.asarray(ds.hf_dataset[i+j]['observation.sensor_time'])[1] for j in range(split['horizon'])])
        assert np.max(np.abs(times-a[1]-nominal))<=.024
        source=int(anchor['observation.source_episode'].item())
        assert (f'episode_{source:06d}' in split['held_out_source_episodes'])==(i in val)
    return {'format':'v3.0','episodes':ds.num_episodes,'frames':len(ds),'all_frames_read_back':True,
            'decoded_video_frames':count,'train_windows':len(train),'validation_windows':len(val),'horizon':split['horizon'],
            'held_out_source_episodes':split['held_out_source_episodes'],'source_disjoint_split':True,
            'research_training_ready':True,'physical_calibration_verified':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo-id',default='local/yumi-block-stack-20261008-training')
    a=p.parse_args()
    print(json.dumps(export(a.prepared,a.root,a.repo_id),indent=2))
