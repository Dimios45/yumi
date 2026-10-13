"""Export the side-by-side alignment view using a captured set of Viser controls."""
import argparse
import json
from pathlib import Path

import align_yam_camera as alignment
import cv2
import mujoco
import numpy as np
import visualize_session_ik as visual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episode', default='episode_000000')
    a = p.parse_args()
    state = json.loads(a.state.read_text())
    manifest = json.loads((a.replay/'manifest.json').read_text())
    report = json.loads((a.replay/f'{a.episode}.json').read_text())
    trajectory = np.load(a.replay/f'{a.episode}-ik.npz')
    world = json.loads((visual.replay.ROOT/'configs/world_camera.json').read_text())
    if state['Camera geometry'].startswith('Fresh'):
        plane = json.loads((visual.replay.ROOT/'artifacts/top-camera-alignment-20261008-plane.json').read_text())
        world['T_table_cam'], world['table_fit'] = plane['T_table_cam'], plane['fit']
    world['base'].update(x_m=state['Forward / back (m): negative = bottom'],
                         y_m=state['Left / right (m): negative = right'],
                         z_m=state['Mounting surface above table (m)'], yaw_deg=state['Base heading (degrees)'])
    photo = cv2.cvtColor(cv2.imread(str(visual.replay.ROOT/'artifacts/top-camera-alignment-20261008.png')), cv2.COLOR_BGR2RGB)
    cam = np.linalg.inv(world['T_table_cam']) @ visual.replay.table_from_base(world['base'])
    wrist = alignment.SessionVideo(Path(manifest['session']))
    solver = visual.replay.DecoupledIKSolver()
    model, data = solver.model, solver.data
    model.geom_matid[:] = -1
    model.geom_rgba[:] = [.12, .72, 1., 1.]
    model.vis.headlight.ambient[:] = .65
    model.vis.headlight.diffuse[:] = .65
    renderer = mujoco.Renderer(model, height=480, width=640)
    process = visual.encoder(a.output, 1280, 480, 30)
    colors = {'actual': (65,219,255), 'target': (245,177,68), 'tracker': (205,125,255)}
    rotations = {'0°': None, '90° clockwise': cv2.ROTATE_90_CLOCKWISE,
                 '180°': cv2.ROTATE_180, '90° counterclockwise': cv2.ROTATE_90_COUNTERCLOCKWISE}
    try:
        for i, device_s in enumerate(trajectory['device_s']):
            data.qpos[:6] = trajectory['joints_rad'][i]
            data.qpos[6:8] = (1-trajectory['gripper_closed'][i])*visual.replay.YAM_FINGER_MAX
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            visual.configure_camera(renderer, cam, world['intrinsics'])
            arm = renderer.render()
            blended = (photo+state['Arm opacity']*arm*(1-photo/255.)).clip(0,255).astype(np.uint8)
            target = trajectory['target'][i]
            poses = {'actual': trajectory['actual'][i], 'target': target,
                     'tracker': target@np.linalg.inv(np.asarray(report['T_tracker_tcp']))}
            uv = {name: visual.replay.project(pose[None,:3,3], cam, world['intrinsics']) for name,pose in poses.items()}
            for start,end,color in [('tracker','target',colors['tracker']), ('target','actual',colors['target'])]:
                if start == 'tracker' and not state['Show T265 origin and tool offset']:
                    continue
                if len(uv[start]) and len(uv[end]):
                    cv2.line(blended, tuple(uv[start][0].astype(int)), tuple(uv[end][0].astype(int)), color, 2, cv2.LINE_AA)
            for name,point in uv.items():
                if len(point) and (name!='tracker' or state['Show T265 origin and tool offset']):
                    cv2.circle(blended, tuple(point[0].astype(int)), 7 if name=='target' else 4, colors[name], 2, cv2.LINE_AA)
            rgb, _ = wrist.read(a.episode, device_s)
            rotation = rotations[state['UMI camera display rotation']]
            if rotation is not None:
                rgb = cv2.rotate(rgb, rotation)
            scale = min(640/rgb.shape[1],480/rgb.shape[0])
            resized = cv2.resize(rgb,(round(rgb.shape[1]*scale), round(rgb.shape[0]*scale)))
            panel = np.zeros_like(photo)
            h,w = resized.shape[:2]
            panel[(480-h)//2:(480+h)//2,(640-w)//2:(640+w)//2] = resized
            for image,title in [(blended,'OVERHEAD / STATIC PHOTO + IK'), (panel,'UMI / RECORDED D405 CAMERA')]:
                cv2.rectangle(image,(0,0),(640,28),(17,25,36),-1)
                cv2.putText(image,title,(10,19),cv2.FONT_HERSHEY_SIMPLEX,.46,(230,240,250),1,cv2.LINE_AA)
            frame = np.concatenate([blended,panel],axis=1)
            process.stdin.write(frame.tobytes())
            if i==len(trajectory['device_s'])//2:
                cv2.imwrite(str(a.output.with_suffix('.png')),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        a.output.with_suffix('.json').write_text(json.dumps({'episode':a.episode,'frames':len(trajectory['device_s']),
            'fps':30,'viewer_settings':state,'world_calibration':world,'source_replay':str(a.replay.resolve())},indent=2)+'\n')
    finally:
        process.stdin.close()
        code = process.wait()
        renderer.close()
        wrist.cap.release()
        if code:
            raise RuntimeError('Video encoding failed')
    print(f'Saved {a.output}',flush=True)


if __name__=='__main__':
    main()
