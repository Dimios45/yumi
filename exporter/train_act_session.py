"""Train ACT relative actions using the checked source-disjoint session split.

Use Karma's LeRobot environment. --smoke-test validates real-data training,
checkpoint reload and prediction with a small CPU model; it does not certify a policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from torch.utils.data import DataLoader, Subset


def pose_to_T(p):
    a,b=p[...,3:6],p[...,6:9]
    x=a/a.norm(dim=-1,keepdim=True)
    y=b-(x*b).sum(-1,keepdim=True)*x
    y=y/y.norm(dim=-1,keepdim=True)
    result=torch.zeros(*p.shape[:-1],4,4,dtype=p.dtype,device=p.device)
    result[...,:3,:3]=torch.stack([x,y,torch.cross(x,y,dim=-1)],dim=-1)
    result[...,:3,3]=p[...,:3]
    result[...,3,3]=1
    return result


def relative_chunk(current,actions):
    rel=torch.linalg.inv(pose_to_T(current.double()))[...,None,:,:]@pose_to_T(actions.double())
    rotation=rel[...,:3,:2].transpose(-1,-2).reshape(*rel.shape[:-2],6)
    return torch.cat([rel[...,:3,3],rotation,actions[...,9:10].double()],dim=-1).float()


def stats(values):
    return {'mean':values.mean(0),'std':values.std(0,unbiased=False).clamp_min(1e-6),
            'min':values.min(0).values,'max':values.max(0).values}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo-id',default='local/yumi-block-stack-20261008-training')
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cuda')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--allow-provisional',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new checkpoint directory')
    contract=json.loads((a.dataset/'meta/umi_contract.json').read_text())
    if contract['contract']!='yumi.act.relative_session.v1':
        raise ValueError('Wrong training contract')
    if not contract['calibration_verified'] and not a.allow_provisional:
        raise ValueError('This research dataset requires --allow-provisional')
    if a.steps<1 or a.batch_size<1:
        raise ValueError('Positive training steps and batch size required')
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu for a pipeline smoke test')
    torch.manual_seed(0)
    torch.set_num_threads(4)
    split=json.loads((a.dataset/'training-split.json').read_text())
    horizon=split['horizon']
    ds=LeRobotDataset(a.repo_id,root=a.dataset,video_backend='pyav',
        delta_timestamps={'action':[i/30 for i in range(horizon)]})
    train=split['train_indices'];val=split['validation_indices']
    # Fit state/action normalization using training sources only.
    states=[];relative=[]
    for index in train:
        current=ds.hf_dataset[index]
        actions=torch.tensor(np.stack([np.asarray(ds.hf_dataset[index+j]['action']) for j in range(horizon)]),dtype=torch.float32)
        ee=torch.tensor(np.asarray(current['observation.ee_pose']),dtype=torch.float32)
        states.append(torch.tensor(np.asarray(current['observation.state']),dtype=torch.float32))
        relative.append(relative_chunk(ee,actions))
    dataset_stats={'observation.state':stats(torch.stack(states)),
        'action':stats(torch.cat(relative)),
        'observation.images.wrist':{
            'mean':torch.tensor([.485,.456,.406])[:,None,None],
            'std':torch.tensor([.229,.224,.225])[:,None,None],
            'min':torch.zeros(3,1,1),'max':torch.ones(3,1,1)}}
    kwargs={'dim_model':128,'dim_feedforward':512,'n_encoder_layers':2,'latent_dim':16} if a.smoke_test else {}
    cfg=ACTConfig(input_features={
        'observation.state':PolicyFeature(FeatureType.STATE,(11,)),
        'observation.images.wrist':PolicyFeature(FeatureType.VISUAL,(3,192,256))},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(10,))},
        chunk_size=horizon,n_action_steps=horizon,device=a.device,
        pretrained_backbone_weights=None,optimizer_lr=1e-4,**kwargs)
    policy=ACTPolicy(cfg).to(a.device)
    pre,post=make_act_pre_post_processors(cfg,dataset_stats=dataset_stats)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4)
    loader=DataLoader(Subset(ds,train),batch_size=a.batch_size,shuffle=True,num_workers=a.workers)
    val_loader=DataLoader(Subset(ds,val),batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    a.output.mkdir(parents=True)
    losses=[]

    def batch(row):
        row['action']=relative_chunk(row['observation.ee_pose'],row['action'])
        if row['action_is_pad'].any():
            raise ValueError('Training window crosses a segment boundary')
        row['observation.images.wrist']=F.interpolate(row['observation.images.wrist'],size=(192,256),mode='bilinear',align_corners=False,antialias=True)
        return pre(row)

    def save(destination):
        policy.save_pretrained(destination)
        pre.save_pretrained(destination)
        post.save_pretrained(destination)

    iterator=iter(loader)
    for step in range(a.steps):
        try:
            row=next(iterator)
        except StopIteration:
            iterator=iter(loader);row=next(iterator)
        policy.train()
        loss,_=policy(batch(row))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step%50==0 or step==a.steps-1:
            print(json.dumps({'step':step+1,'loss':losses[-1]}),flush=True)
        if (step+1)%a.save_every==0:
            save(a.output/f'step_{step+1:06d}')
    policy.eval()
    validation=[]
    with torch.inference_mode():
        for row in val_loader:
            loss,_=policy(batch(row))
            validation.append(float(loss))
            if a.smoke_test:
                break
    save(a.output/'last')
    restored=ACTPolicy.from_pretrained(a.output/'last').to(a.device)
    restored.eval()
    with torch.inference_mode():
        row=batch(next(iter(val_loader)))
        predicted=restored.predict_action_chunk(row)
        assert predicted.shape==(len(row['observation.state']),horizon,10) and torch.isfinite(predicted).all()
    result={'steps':a.steps,'smoke_test':a.smoke_test,'last_training_loss':losses[-1],
        'validation_loss':float(np.mean(validation)),'validation_batches':len(validation),
        'train_windows':len(train),'validation_windows':len(val),
        'held_out_source_episodes':split['held_out_source_episodes'],'checkpoint_reload_and_prediction_passed':True,
        'normalization_fitted_on_training_sources_only':True,'hardware_validated':False,
        'calibration_verified':contract['calibration_verified'],'source_dataset':str(a.dataset.resolve()),
        'camera_resize':[192,256],'gripper_contract':contract['gripper_semantics']}
    (a.output/'training-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
