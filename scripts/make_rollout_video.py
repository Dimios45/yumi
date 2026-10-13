#!/usr/bin/env python3
"""Blog video of a policy rollout: the training demonstration next to the robot running the policy.

Inputs are an `infer_act_relative.py live --record DIR --log RUN.json` recording (DIR/wrist.mp4 from the
wrist D405, DIR/top.mp4 from the top D435, each with a .json of frame capture times) and one episode of the
LeRobot training dataset. Writes, into DIR:

  demo.mp4     the training episode (UMI handheld gripper, wrist camera)
  rollout.mp4  the robot: top camera with the wrist D405 beside it
  blog.mp4     both merged at 1920x1080: demo and robot wrist on the left, top camera on the right

The rollout is trimmed to the policy's run (from --lead s before the first observation to the last
command), shown at --speedup with a clock, and the policy's gripper width from the run log.

Run from ~/karma:
  uv run python ../yumi/scripts/make_rollout_video.py ../yumi/artifacts/rollout-01 \\
    --log ../yumi/artifacts/rollout-01.json --episode 44
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
BG, PANEL, TEXT, DIM, ACCENT = (17, 19, 24), (30, 33, 41), (236, 238, 242), (150, 156, 168), (95, 200, 120)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False):
    try:
        return ImageFont.truetype(BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def probe_size(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height', '-of', 'csv=p=0', str(path)], capture_output=True, text=True, check=True)
    w, h = out.stdout.strip().split(',')[:2]
    return int(w), int(h)


class Frames:
    """Sequential RGB frames of a video with their times; at(t) returns the latest frame at or before t."""

    def __init__(self, path, times=None, start=None, end=None):
        self.size = probe_size(path)
        cmd = ['ffmpeg', '-loglevel', 'error']
        if start is not None:
            cmd += ['-ss', f'{start:.4f}', '-to', f'{end:.4f}']
        self.proc = subprocess.Popen(cmd+['-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                     stdout=subprocess.PIPE)
        self.times = times
        self.i, self.frame, self.next = -1, None, self._read()

    def _read(self):
        n = self.size[0]*self.size[1]*3
        buf = self.proc.stdout.read(n)
        return np.frombuffer(buf, np.uint8).reshape(self.size[1], self.size[0], 3) if len(buf) == n else None

    def time(self, i):
        if self.times is None:
            return i/FPS
        return self.times[i] if i < len(self.times) else np.inf

    def at(self, t):
        while self.next is not None and self.time(self.i+1) <= t:
            self.frame, self.next, self.i = self.next, self._read(), self.i+1
        return self.frame if self.frame is not None else self.next

    @property
    def done(self):
        return self.next is None

    def close(self):
        self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()


class Writer:
    def __init__(self, path, size):
        self.path = path
        self.proc = subprocess.Popen(
            ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}',
             '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)

    def write(self, image):
        self.proc.stdin.write(np.asarray(image.convert('RGB')).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        print(f'wrote {self.path}')


def fit(rgb, w, h):
    return Image.fromarray(rgb).resize((w, h), Image.LANCZOS) if rgb is not None else Image.new('RGB', (w, h), PANEL)


def labelled(canvas, draw, rgb, box, title, subtitle=''):
    """Paste a camera frame at box=(x, y, w, h) with a title above it."""
    x, y, w, h = box
    draw.text((x, y-34), title, font=font(22, True), fill=TEXT)
    if subtitle:
        tw = draw.textlength(title, font=font(22, True))
        draw.text((x+tw+12, y-30), subtitle, font=font(17), fill=DIM)
    draw.rounded_rectangle((x-2, y-2, x+w+1, y+h+1), 6, fill=PANEL)
    canvas.paste(fit(rgb, w, h), (x, y))


def chip(draw, xy, text, color=TEXT, size=20):
    f = font(size, True)
    x, y = xy
    w = draw.textlength(text, font=f)
    draw.rounded_rectangle((x, y, x+w+20, y+size+14), 8, fill=(0, 0, 0, 170))
    draw.text((x+10, y+6), text, font=f, fill=color)
    return x+w+28


def gripper_bar(draw, xy, width_mm, max_mm=60.):
    x, y = xy
    draw.text((x, y), 'gripper', font=font(17), fill=DIM)
    x0, w = x+80, 220
    draw.rounded_rectangle((x0, y+3, x0+w, y+19), 8, fill=PANEL)
    if width_mm is not None:
        draw.rounded_rectangle((x0, y+3, x0+max(16, w*min(width_mm, max_mm)/max_mm), y+19), 8, fill=ACCENT)
        draw.text((x0+w+12, y), f'{width_mm:.0f} mm', font=font(17, True), fill=TEXT)


def load_episode(dataset, episode):
    import pandas as pd

    meta = json.loads((dataset/'meta/info.json').read_text())
    eps = pd.concat(pd.read_parquet(f) for f in sorted((dataset/'meta/episodes').rglob('*.parquet')))
    row = eps[eps.episode_index == episode].iloc[0]
    key = 'observation.images.wrist'
    path = dataset/meta['video_path'].format(video_key=key, chunk_index=int(row[f'videos/{key}/chunk_index']),
                                             file_index=int(row[f'videos/{key}/file_index']))
    return path, float(row[f'videos/{key}/from_timestamp']), float(row[f'videos/{key}/to_timestamp'])


def run_window(log_path, wrist_t, lead, tail):
    """Policy start/end on time.monotonic() from the run log, and the policy's gripper width over time."""
    if log_path is None:
        return wrist_t[0], wrist_t[-1], np.zeros(0), np.zeros(0)
    chunks = [c for c in json.loads(Path(log_path).read_text())['chunks'] if isinstance(c, dict)]
    steps = sorted((st['t'], st.get('width_pred_mm')) for c in chunks for st in c.get('steps', []))
    t_obs = [c['t_obs'] for c in chunks if 't_obs' in c]
    t0 = (min(t_obs) if t_obs else steps[0][0] if steps else wrist_t[0])-lead
    t1 = (steps[-1][0] if steps else wrist_t[-1])+tail
    st = [(t, w) for t, w in steps if w is not None]
    return max(t0, wrist_t[0]), min(t1, wrist_t[-1]), np.array([t for t, _ in st]), np.array([w for _, w in st])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('record', type=Path, help='--record DIR of the run')
    p.add_argument('--log', type=Path, help='--log JSON of the same run (trims to the policy and adds the gripper)')
    p.add_argument('--dataset', type=Path, default=ROOT/'data/lerobot/yumi-umi-block-place-v2')
    p.add_argument('--episode', type=int, default=44, help='training episode to show (44 = held out)')
    p.add_argument('--speedup', type=float, default=1., help='rollout playback speed (the demo plays at 1x)')
    p.add_argument('--lead', type=float, default=1., help='seconds shown before the first observation')
    p.add_argument('--tail', type=float, default=1.5, help='seconds shown after the last command')
    p.add_argument('--title', default='ACT + CLIP · pick and place the green block')
    args = p.parse_args()

    wrist_t = np.array(json.loads((args.record/'wrist.json').read_text())['t'])
    has_top = (args.record/'top.mp4').exists()
    top_t = np.array(json.loads((args.record/'top.json').read_text())['t']) if has_top else None
    t0, t1, grip_t, grip_w = run_window(args.log, wrist_t, args.lead, args.tail)
    duration = (t1-t0)/args.speedup
    ep_path, ep_from, ep_to = load_episode(args.dataset, args.episode)
    demo_len = ep_to-ep_from
    speed = f'{args.speedup:g}x' if args.speedup != 1 else 'real time'
    print(f'rollout {t1-t0:.1f} s (shown {duration:.1f} s, {speed}); demo episode {args.episode}: {demo_len:.1f} s')

    def width_at(t):
        if not len(grip_t) or t < grip_t[0]:
            return None
        return float(grip_w[min(np.searchsorted(grip_t, t), len(grip_w)-1)])

    def sources():
        return (Frames(ep_path, start=ep_from, end=ep_to), Frames(args.record/'wrist.mp4', wrist_t),
                Frames(args.record/'top.mp4', top_t) if has_top else None)

    # demo.mp4: the training episode alone
    demo = Frames(ep_path, start=ep_from, end=ep_to)
    out = Writer(args.record/'demo.mp4', (640, 480))
    while not demo.done:
        demo.at(demo.time(demo.i+1))
        img = Image.fromarray(demo.frame)
        draw = ImageDraw.Draw(img, 'RGBA')
        chip(draw, (14, 14), f'Training demo · episode {args.episode}', size=17)
        out.write(img)
    demo.close()
    out.close()

    # rollout.mp4 (top + wrist side by side, 1920x720) and blog.mp4 (1920x1080), one pass over the sources
    demo, wrist, top = sources()
    roll = Writer(args.record/'rollout.mp4', (1920, 720))
    blog = Writer(args.record/'blog.mp4', (1920, 1080))
    n = int(np.ceil(max(duration, demo_len)*FPS))
    for i in range(n):
        shown = i/FPS
        t = t0+min(shown, duration)*args.speedup
        w_rgb, top_rgb = wrist.at(t), top.at(t) if top is not None else None
        d_rgb = demo.at(min(shown, demo_len))
        width = width_at(t)
        clock = f't = {min(shown, duration)*args.speedup:4.1f} s'

        # rollout.mp4: top camera 1280x720 | wrist 640x480 with the overlays under it
        img = Image.new('RGB', (1920, 720), BG)
        draw = ImageDraw.Draw(img, 'RGBA')
        img.paste(fit(top_rgb if top_rgb is not None else w_rgb, 1280, 720), (0, 0))
        img.paste(fit(w_rgb, 640, 480), (1280, 0))
        x = chip(draw, (16, 16), 'Top camera' if top_rgb is not None else 'Wrist D405', size=18)
        chip(draw, (x, 16), clock+(f'  ·  {speed}' if args.speedup != 1 else ''), color=ACCENT, size=18)
        chip(draw, (1296, 16), 'Wrist D405 · policy input', size=18)
        draw.text((1300, 510), args.title, font=font(22, True), fill=TEXT)
        draw.text((1300, 548), 'YAM arm running the policy, trained on UMI handheld demos', font=font(16), fill=DIM)
        gripper_bar(draw, (1300, 600), width)
        roll.write(img)

        # blog.mp4: training demo + robot wrist on the left, top camera on the right
        img = Image.new('RGB', (1920, 1080), BG)
        draw = ImageDraw.Draw(img, 'RGBA')
        labelled(img, draw, d_rgb, (40, 92, 600, 450), 'Training data', f'UMI handheld · episode {args.episode}')
        if shown > demo_len:  # demo finished: hold its last frame, dimmed
            draw.rectangle((40, 92, 640, 542), fill=(0, 0, 0, 120))
            draw.text((290, 300), 'end', font=font(22, True), fill=DIM)
        labelled(img, draw, w_rgb, (40, 600, 600, 450), 'Robot · wrist D405', 'what the policy sees')
        big = top_rgb if top_rgb is not None else w_rgb
        labelled(img, draw, big, (680, 92, 1200, 675), 'Robot · top camera' if top_rgb is not None else 'Robot',
                 'YAM running the policy')
        x = chip(draw, (696, 108), clock, color=ACCENT, size=20)
        if args.speedup != 1:
            chip(draw, (x, 108), speed, size=20)
        draw.text((680, 800), args.title, font=font(30, True), fill=TEXT)
        draw.text((680, 846), 'ACT policy trained on handheld UMI demonstrations, deployed on a YAM arm',
                  font=font(19), fill=DIM)
        gripper_bar(draw, (680, 900), width)
        blog.write(img)
    for f in (demo, wrist, top):
        if f is not None:
            f.close()
    roll.close()
    blog.close()


if __name__ == '__main__':
    raise SystemExit(main())
