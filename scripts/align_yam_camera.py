"""Interactive camera/base alignment preview; reads saved IK, never robot hardware."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import cv2
import mujoco
import numpy as np
import viser
import visualize_session_ik as visual


class SessionVideo:
    """Decode wrist video at the nearest actual image timestamp in each episode."""
    def __init__(self, root):
        self.images = {}
        offset = 0
        for ep in sorted((root/'extras').glob('episode_*')):
            rows = [json.loads(line) for line in (ep/'samples.jsonl').read_text().splitlines()]
            rows = [row for row in rows if row['kind'] == 'image']
            self.images[ep.name] = (np.array([row['device_s'] for row in rows]), offset)
            offset += len(rows)
        files = sorted((root/'videos/observation.images.d405').glob('chunk-*/*.mp4'))
        if len(files) != 1:
            raise ValueError('This viewer currently expects one concatenated D405 video file')
        self.cap = cv2.VideoCapture(str(files[0]))
        if not self.cap.isOpened() or int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) != offset:
            raise ValueError('Wrist video frame count does not match session image rows')
        self.last_index, self.last_frame = -1, None

    def read(self, episode, device_s):
        times, offset = self.images[episode]
        local = int(np.abs(times-device_s).argmin())
        index = offset+local
        if index != self.last_index:
            if index != self.last_index+1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self.cap.read()
            if not ok:
                raise ValueError(f'Cannot decode wrist frame {index}')
            self.last_index, self.last_frame = index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.last_frame, float(times[local]-device_s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--photo', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--plane', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=8082)
    p.add_argument('--session', type=Path, help='Raw LeRobot session; defaults to the replay manifest source')
    a = p.parse_args()
    original = json.loads((visual.replay.ROOT/'configs/world_camera.json').read_text())
    plane = json.loads(a.plane.read_text())
    photo = cv2.cvtColor(cv2.imread(str(a.photo)), cv2.COLOR_BGR2RGB)
    episodes = {p.stem.replace('-ik', ''): np.load(p) for p in sorted(a.replay.glob('*-ik.npz'))}
    reports = {name: json.loads((a.replay/f'{name}.json').read_text()) for name in episodes}
    session = a.session or Path(json.loads((a.replay/'manifest.json').read_text())['session'])
    wrist = SessionVideo(session)
    initial_base = json.loads(a.output.read_text())['base'] if a.output.exists() else {
        'x_m': -.025, 'y_m': -.34, 'z_m': 0., 'yaw_deg': 0.}
    solver = visual.replay.DecoupledIKSolver()
    model, data = solver.model, solver.data
    model.geom_matid[:] = -1
    model.geom_rgba[:] = [.12, .72, 1, 1]
    model.vis.headlight.ambient[:] = .65
    model.vis.headlight.diffuse[:] = .65
    renderer = mujoco.Renderer(model, height=480, width=640)
    server = viser.ViserServer(host='0.0.0.0', port=a.port, label='YAM camera alignment')
    server.gui.configure_theme(control_width='large', brand_color=(39, 135, 173))
    server.gui.add_markdown('## Right YAM · camera alignment\nFresh static photo. Adjust the base in metres; **this is a provisional visual fit**.')
    picture = server.gui.add_image(np.concatenate([photo, photo], axis=1),
                                   label='Overhead + YAM  |  recorded UMI wrist camera', format='jpeg', jpeg_quality=90)
    wrist_status = server.gui.add_markdown('')
    wrist_rotation = server.gui.add_dropdown('UMI camera display rotation', options=['0°', '90° clockwise', '180°', '90° counterclockwise'])
    opacity = server.gui.add_slider('Arm opacity', min=0, max=1, step=.01, initial_value=.28)
    mode = server.gui.add_dropdown('Robot pose', options=['Park (assumed q=0)', 'Recorded IK'], initial_value='Recorded IK')
    episode = server.gui.add_dropdown('Episode', options=list(episodes))
    scrub = server.gui.add_slider('Time (s)', min=0, max=20, step=.0333333, initial_value=0)
    playing = server.gui.add_checkbox('Play recording', False)
    show_tracker = server.gui.add_checkbox('Show T265 origin and tool offset', True)
    ee_status = server.gui.add_markdown('')
    server.gui.add_markdown('**Cyan: actual YAM EE between fingertips. Orange: transformed UMI EE target. Purple: T265 origin.** The purple-to-orange line is the tracker-to-tool offset; orange-to-cyan is IK error.')
    with server.gui.add_folder('Base placement', expand_by_default=True):
        x = server.gui.add_slider('Forward / back (m): negative = bottom', min=-.3, max=.3, step=.001, initial_value=initial_base['x_m'])
        y = server.gui.add_slider('Left / right (m): negative = right', min=-.6, max=.1, step=.001, initial_value=initial_base['y_m'])
        z = server.gui.add_slider('Mounting surface above table (m)', min=-.15, max=.2, step=.001, initial_value=initial_base['z_m'])
        yaw = server.gui.add_slider('Base heading (degrees)', min=-180, max=180, step=.5, initial_value=initial_base['yaw_deg'])
        camera = server.gui.add_dropdown('Camera geometry', options=['Saved calibration', 'Fresh depth plane (estimated)'])
        reset = server.gui.add_button('Reset to original saved base')
        save = server.gui.add_button('Save candidate alignment and preview')
        saved = server.gui.add_markdown('')
    server.gui.add_markdown('### To measure accurately\n1. Make the complete right mounting plate visible.\n2. Measure its centre relative to the point directly below the camera: forward/back and left/right, on the table plane.\n3. Measure mounting-surface height and base heading.\n4. Compare using known joint angles; q=0 here is only an assumption.\n\nFor a precise camera/base transform, use visible, known base-plate hole centres or a rigidly attached calibrated target, then verify on additional points/poses. A new base alignment does not calibrate the handheld tracker-to-robot transform.')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/table', width=2, height=2)
    markers = {}
    colors = {'actual': (65, 219, 255), 'target': (245, 177, 68), 'tracker': (205, 125, 255)}
    labels = {'actual': 'YAM EE (FK)', 'target': 'UMI EE target', 'tracker': 'T265 origin'}
    for name, color in colors.items():
        frame = server.scene.add_frame(f'/poses/{name}', axes_length=.045, axes_radius=.0015)
        server.scene.add_icosphere(f'/poses/{name}/point', radius=.005, color=color)
        server.scene.add_label(f'/poses/{name}/label', text=labels[name], position=(0, 0, .025))
        markers[name] = frame
    offset_line = server.scene.add_line_segments('/tracker_to_ee', points=np.zeros((1,2,3), np.float32), colors=colors['tracker'], line_width=3)
    error_line = server.scene.add_line_segments('/ik_error', points=np.zeros((1,2,3), np.float32), colors=colors['target'], line_width=3)
    @server.on_client_connect
    def _(client):
        client.camera.position = (1.1, -.9, .8)
        client.camera.look_at = (.25, 0., .25)
    handles = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        handles[gid] = server.scene.add_mesh_simple(f'/yam/{gid}', vertices=model.mesh_vert[va:va+vn],
            faces=model.mesh_face[fa:fa+fn], color=(45, 185, 245), opacity=.28)
    def candidate():
        doc = json.loads(json.dumps(original))
        if camera.value.startswith('Fresh'):
            doc['T_table_cam'] = plane['T_table_cam']
            doc['table_fit'] = plane['fit']
        doc['base'].update(x_m=x.value, y_m=y.value, z_m=z.value, yaw_deg=yaw.value)
        doc.update(registration_verified=False, note='Unverified interactive visual alignment; photo does not supply joint angles.',
                   alignment_photo=str(a.photo.resolve()))
        return doc
    @reset.on_click
    def _(_event):
        b = original['base']
        x.value, y.value, z.value, yaw.value = b['x_m'], b['y_m'], b['z_m'], b['yaw_deg']
    @save.on_click
    def _(_event):
        a.output.write_text(json.dumps(candidate(), indent=2)+'\n')
        cv2.imwrite(str(a.output.with_suffix('.png')), cv2.cvtColor(picture.image, cv2.COLOR_RGB2BGR))
        saved.content = f'Saved candidate: `{a.output.name}`. Original calibration unchanged.'
    last = None
    previous = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if playing.value and mode.value == 'Recorded IK' and server.get_clients():
                scrub.value = (scrub.value+min(now-previous, .2)) % float(episodes[episode.value]['time_s'][-1])
            previous = now
            key = (x.value, y.value, z.value, yaw.value, camera.value, mode.value, episode.value, scrub.value, opacity.value, show_tracker.value, wrist_rotation.value)
            if key != last:
                doc = candidate()
                data.qpos[:] = 0
                if mode.value == 'Recorded IK':
                    ep = episodes[episode.value]
                    i = min(len(ep['time_s'])-1, int(scrub.value*30))
                    data.qpos[:6] = ep['joints_rad'][i]
                    data.qpos[6:8] = (1-ep['gripper_closed'][i])*visual.replay.YAM_FINGER_MAX
                mujoco.mj_forward(model, data)
                actual = np.eye(4)
                actual[:3, 3] = data.site_xpos[solver.site_id]
                actual[:3, :3] = data.site_xmat[solver.site_id].reshape(3,3)
                poses = {'actual': actual}
                recorded = mode.value == 'Recorded IK'
                if recorded:
                    target = ep['target'][i]
                    tracker = target @ np.linalg.inv(np.asarray(reports[episode.value]['T_tracker_tcp']))
                    poses.update(target=target, tracker=tracker)
                    offset_line.points = np.array([[tracker[:3,3], target[:3,3]]], np.float32)
                    error_line.points = np.array([[target[:3,3], actual[:3,3]]], np.float32)
                    pos_err, rot_err = visual.replay.pose_error(target, actual)
                    ee_status.content = f'EE error: **{pos_err:.2f} mm / {rot_err:.2f}°** · '+('valid tracker sample' if ep['valid'][i] else 'invalid tracker sample — arm holds')
                else:
                    ee_status.content = 'Park preview: actual model EE shown; recorded target hidden.'
                offset_line.visible = recorded and show_tracker.value
                error_line.visible = recorded
                for name, marker in markers.items():
                    marker.visible = name in poses and (name != 'tracker' or show_tracker.value)
                    if name in poses:
                        marker.position = poses[name][:3,3]
                        marker.wxyz = visual.replay.wxyz(poses[name][:3,:3])
                cam = np.linalg.inv(doc['T_table_cam']) @ visual.replay.table_from_base(doc['base'])
                renderer.update_scene(data)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                visual.configure_camera(renderer, cam, doc['intrinsics'])
                arm = renderer.render()
                blended = (photo+opacity.value*arm*(1-photo/255.)).clip(0,255).astype(np.uint8)
                uv = {name: visual.replay.project(pose[None,:3,3], cam, doc['intrinsics']) for name, pose in poses.items()}
                for start, end, color in [('tracker', 'target', colors['tracker']), ('target', 'actual', colors['target'])]:
                    if start == 'tracker' and not show_tracker.value:
                        continue
                    if start in uv and end in uv and len(uv[start]) and len(uv[end]):
                        cv2.line(blended, tuple(uv[start][0].astype(int)), tuple(uv[end][0].astype(int)), color, 2, cv2.LINE_AA)
                for name, point in uv.items():
                    if len(point) and markers[name].visible:
                        u, v = point[0].astype(int)
                        cv2.circle(blended, (u,v), 7 if name == 'target' else 4, colors[name], 2, cv2.LINE_AA)
                ep_video = episodes[episode.value]
                video_index = min(len(ep_video['time_s'])-1, int(scrub.value*30))
                wrist_rgb, skew = wrist.read(episode.value, ep_video['device_s'][video_index])
                rotations = {'0°': None, '90° clockwise': cv2.ROTATE_90_CLOCKWISE,
                             '180°': cv2.ROTATE_180, '90° counterclockwise': cv2.ROTATE_90_COUNTERCLOCKWISE}
                rotation = rotations[wrist_rotation.value]
                if rotation is not None:
                    wrist_rgb = cv2.rotate(wrist_rgb, rotation)
                panel = np.zeros_like(photo)
                scale = min(640/wrist_rgb.shape[1], 480/wrist_rgb.shape[0])
                resized = cv2.resize(wrist_rgb, (round(wrist_rgb.shape[1]*scale), round(wrist_rgb.shape[0]*scale)))
                h, w = resized.shape[:2]
                panel[(480-h)//2:(480+h)//2, (640-w)//2:(640+w)//2] = resized
                for image, title in [(blended, 'OVERHEAD / STATIC PHOTO + IK'), (panel, 'UMI / RECORDED D405 CAMERA')]:
                    cv2.rectangle(image, (0,0), (640,28), (17,25,36), -1)
                    cv2.putText(image, title, (10,19), cv2.FONT_HERSHEY_SIMPLEX, .46, (230,240,250), 1, cv2.LINE_AA)
                picture.image = np.concatenate([blended, panel], axis=1)
                wrist_status.content = f'UMI camera: **{episode.value} · {scrub.value:.2f} s** · nearest image offset {skew*1000:+.1f} ms'
                for gid, mesh in handles.items():
                    mesh.position = data.geom_xpos[gid].copy()
                    mesh.wxyz = visual.replay.wxyz(data.geom_xmat[gid].reshape(3,3))
                    mesh.opacity = opacity.value
                if last is None and not a.output.exists():
                    a.output.write_text(json.dumps(doc, indent=2)+'\n')
                    cv2.imwrite(str(a.output.with_suffix('.png')), cv2.cvtColor(picture.image, cv2.COLOR_RGB2BGR))
                last = key
            time.sleep(.05)
    finally:
        wrist.cap.release()
        renderer.close()
        server.stop()


if __name__ == '__main__':
    main()
