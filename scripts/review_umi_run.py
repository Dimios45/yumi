"""Offline adapter for existing yam_motion planner/streamer and replay_yam viewer."""
import sys,json,argparse,threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
sys.path.insert(0,str(Path('scripts').resolve()))
import replay_yam as r
from yam_motion import PoseIK,Planner,JointStreamer
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('capture',type=Path)
parser.add_argument('--port',type=int,default=8082)
parser.add_argument('--no-viewer',action='store_true')
parser.add_argument('--decision-file',type=Path,help='Enable Viser accept/retry controls for the collection loop')
cli=parser.parse_args()
if cli.no_viewer and cli.decision_file:
 parser.error('--decision-file requires the interactive viewer')
root=cli.capture.resolve()
if not (root/'COMPLETE').exists(): raise SystemExit('Capture is incomplete')
profile_path=Path('configs/yam.umi-start.json')
profile=json.loads(profile_path.read_text())
meta,images,times,grid,tracker,valid=r.load_capture(root,3,.025,30,diagnostic=True)
tracking_ok=bool(valid.all())
if not tracking_ok: print('Low-confidence tracking: diagnostic preview only; acceptance disabled.',flush=True)
# Anchor at the first recorded pose even when tracking is unverified.
anchor_valid=np.ones(len(valid),dtype=bool)
config=json.loads(Path('configs/umi.review-collection.json').read_text())
ext=np.array(profile['T_tracker_tcp']);q0=np.array(profile['start_joints_rad']);ik=PoseIK(q0)
home=ik.fk(q0);anchor_home=ik.fk(profile['anchor_joints_rad'])
targets=r.map_targets(tracker@ext,anchor_valid,'gravity',anchor_home)
plan=Planner(ik,transit_m=.06,lift_m=.06,floor_z=.01,max_miss_mm=25).plan(q0,targets,times+.5,0.)
plan_complete=bool(plan['cut']==len(plan['q']))
if not plan_complete: print('Planner rejected targets: diagnostic preview only; acceptance disabled.',flush=True)
widths=r.measure_widths(root,images,meta,config['markers'])
image_times=np.array([x['device_s'] for x in images]);nearest=np.abs(grid[:,None]-image_times[None,:]).argmin(1)
width_grid=np.array([np.nan if widths[j] is None else widths[j] for j in nearest]);start_closed=0.5
closed=[];last=start_closed
for w in width_grid:
 if np.isfinite(w):last=float(np.clip(1-w/.095,0,1))
 closed.append(last)
stream=JointStreamer(lambda q,g:None,200,q0,start_closed,[1.5]*3+[3.]*3,[6.]*3+[12.]*3)
stream.t_knots=np.r_[0.,plan['times']];stream.q_knots=np.vstack([q0,plan['q']]);stream.g_knots=np.r_[start_closed,np.full(plan['transit'],start_closed),closed]
knots=np.concatenate([home[None],plan['targets']]);slerp=Slerp(stream.t_knots,Rotation.from_matrix(knots[:,:3,:3]))
start=float(plan['times'][plan['transit']]);end=float(plan['times'][-1]);display=np.arange(0,end,1/30)
samples=[];tick=0
for t in display:
 while (tick+1)/200<=t+1e-9:tick+=1;stream.step(tick/200)
 target=np.eye(4);target[:3,3]=[np.interp(t,stream.t_knots,knots[:,j,3]) for j in range(3)];target[:3,:3]=slerp(t).as_matrix();actual=ik.fk(stream.qpos)
 pe,re=r.pose_error(target,actual);i=int(np.clip(round((t-start)*30),0,len(times)-1));approach=t<start
 samples.append({'time':float(t),'valid':bool(approach or valid[i]),'q':stream.qpos,'gripper':stream.gripper,'width':None if approach or not np.isfinite(width_grid[i]) else float(width_grid[i]),'target':target,'actual':actual,'position_error_mm':pe,'rotation_error_deg':re,'status':'ok' if (approach or valid[i]) and pe<=10 and re<=5 else 'miss','image':None if approach else int(nearest[i])})
assert np.array_equal(samples[0]['q'],q0)
recorded=[s for s in samples if s['time']>=start];pe=np.array([s['position_error_mm'] for s in recorded]);re=np.array([s['rotation_error_deg'] for s in recorded]);fraction=float(np.mean((pe<=10)&(re<=5)))
summary=f'**Existing smooth planner + velocity/acceleration-limited streamer**\n\nSaved UMI task start → approach for **{start:.2f}s** → {root.name}.\n\nEstablished TCP axes; gravity alignment at the earlier ready-pose workspace.\n\nRecorded portion: **{fraction:.1%}** within 10 mm / 5°.\n\nPosition p95 / max: **{np.percentile(pe,95):.1f} / {pe.max():.1f} mm**.\n\nRotation p95 / max: **{np.percentile(re,95):.1f} / {re.max():.1f}°**.\n\nMissing jaw measurements hold the previous visual command; they remain invalid. Physical TCP calibration is provisional.'
report={'duration_s':float(display[-1]),'samples':len(samples),'mapping':'gravity','initial_joints_rad':q0.tolist(),'T_tracker_tcp':ext.tolist(),'anchor_xyz_m_yaw_deg':list(r.start_anchor(tracker@ext,anchor_valid,anchor_home)),'approach_duration_s':start,'recorded_tracked_fraction':fraction,'position_error_mm':{'max':float(pe.max()),'p95':float(np.percentile(pe,95))},'rotation_error_deg':{'max':float(re.max()),'p95':float(np.percentile(re,95))},'comparison_summary':summary,'hardware_connection':False,'planner':'existing scripts/yam_motion.py PoseIK + Planner + JointStreamer','physical_registration_verified':False}
# Fingerprint all raw inputs and configs; acceptance must refer to this exact review.
from umi_review_state import fingerprint
report['fingerprint']=fingerprint(root)
report['profile']=profile
report['tracking_valid_fraction']=float(np.mean(valid))
report['diagnostic_tracking']=not tracking_ok
report['planner_complete']=plan_complete
report['width_valid_fraction']=float(np.mean(np.isfinite(width_grid)))
report['review_gate_passed']=bool(tracking_ok and plan_complete and fraction>=.95 and report['width_valid_fraction']>=.95 and pe.max()<=25 and re.max()<=10 and plan['floored']==0)
report['review_gate']='Confidence 3 and no tracking gaps throughout; complete plan; >=95% poses within 10mm/5deg; max 25mm/10deg; >=95% measured widths; no floor corrections'
report['status']='pending_human_review'
report['review_reasons']=[]
if not tracking_ok: report['review_reasons'].append(f'Tracking valid at confidence 3 for {np.mean(valid):.1%}; low-confidence poses are diagnostic only.')
if not plan_complete: report['review_reasons'].append('Planner rejected targets; simulation is diagnostic only.')
if report['review_reasons']:
 summary='**DIAGNOSTIC PREVIEW — NOT ACCEPTABLE**\n\n'+' '.join(report['review_reasons'])+'\n\n'+summary
 report['comparison_summary']=summary
(root/'umi-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(summary,flush=True)
print(f"Width coverage: {report['width_valid_fraction']:.1%}; acceptance gate: {report['review_gate_passed']}",flush=True)
if cli.no_viewer: raise SystemExit(0)
args=SimpleNamespace(capture=root,host='0.0.0.0',port=cli.port,rate=30,ik_hz=200,mu=.005,mapping='gravity',initial_joints=q0.tolist(),mount_tweak=[0,0,0],tcp_offset=ext[:3,3],fixed_smooth_comparison=True)
cap=(meta,images,times,grid,tracker,valid,image_times,widths)
state={'samples':samples,'report':report,'tcp_source':'Established replay axes; physical TCP still provisional','inner':None}
if cli.decision_file:
 from umi_review_controls import add_review_controls
 decision=cli.decision_file.resolve()
 if decision.is_relative_to(root): raise SystemExit('Decision file must be outside the capture directory')
 state['stop_event']=threading.Event()
 state['on_viewer_ready']=lambda server:add_review_controls(server,root,decision,report,state['stop_event'])
r.serve(args,cap,state,ext[:3,:3],None,None)
