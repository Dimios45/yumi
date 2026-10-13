"""ACT + CLIP, UMI relative actions, on vruga/pick-place-the-green-block (45 episodes).

Trains on every episode except --val-eps (default: episode 44 = second-session episode 23, held out), logs
val L1 every --eval-every steps and keeps the best-val checkpoint ("best") + periodic ones.
Optional --max-gap drops episodes whose longest interpolated gripper-width gap exceeds that many frames.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_relative import IMG, build_act_clip  # noqa: E402  (puts pydeps on sys.path)
from train_act_relative import relative_action_chunk  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402

FPS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/sra/vrushtee/yumi_act/data_v2")
    ap.add_argument("--repo-id", default="vruga/pick-place-the-green-block")
    ap.add_argument("--out", default="/home/sra/vrushtee/yumi_act/outputs/v2_act_clip_relative")
    ap.add_argument("--max-gap", type=int, default=None, help="drop episodes with a longer width gap (default: keep all)")
    ap.add_argument("--val-eps", type=int, nargs="*", default=[44])
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads((Path(args.root) / "meta" / "umi_export.json").read_text())
    names = info["episodes"]  # dataset episode_index i  <->  names[i]
    keep = [i for i, n in enumerate(names)
            if args.max_gap is None or info["per_episode"][n]["longest_width_gap_frames"] <= args.max_gap]
    val_eps = [e for e in args.val_eps if e in keep]
    train_eps = [e for e in keep if e not in val_eps]
    print("dropped:", [names[i] for i in range(len(names)) if i not in keep])
    print("train eps:", train_eps, "val eps:", val_eps, flush=True)

    delta = {"action": [i / FPS for i in range(args.chunk)]}
    mk = lambda eps: LeRobotDataset(args.repo_id, root=args.root, episodes=eps, video_backend="pyav",  # noqa: E731
                                    delta_timestamps=delta)
    ds, vds = mk(train_eps), mk(val_eps)
    feats = dataset_to_policy_features(ds.meta.features)
    cfg = ACTConfig(
        input_features={k: feats[k] for k in (IMG, "observation.state")},
        output_features={"action": feats["action"]},
        chunk_size=args.chunk, n_action_steps=args.chunk, pretrained_backbone_weights=None,
        optimizer_lr=args.lr, device="cuda")

    # normalisation stats from training episodes only
    df = pd.read_parquet(Path(args.root) / "data/chunk-000/file-000.parquet")
    df = df[df.episode_index.isin(train_eps)]
    chunks, poses, states = [], [], []
    for _, ep in df.groupby("episode_index"):
        act = torch.tensor(np.stack(ep["action"].values), dtype=torch.float32)
        idx = (torch.arange(len(ep))[:, None] + torch.arange(args.chunk)[None]).clamp(max=len(ep) - 1)
        chunks.append(act[idx])
        poses.append(torch.tensor(np.stack(ep["observation.ee_pose"].values), dtype=torch.float32))
        states.append(torch.tensor(np.stack(ep["observation.state"].values), dtype=torch.float32))
    rel = relative_action_chunk(torch.cat(poses), torch.cat(chunks)).reshape(-1, 10)
    st = torch.cat(states)

    def f(x):
        return {"mean": x.mean(0), "std": x.std(0).clamp_min(1e-6), "min": x.min(0).values, "max": x.max(0).values}
    stats = {"action": f(rel), "observation.state": f(st),
             IMG: {"mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
                   "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)}}

    policy = build_act_clip(cfg, stats[IMG]).cuda().train()
    pre, post = make_act_pre_post_processors(cfg, dataset_stats=stats)
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in policy.named_parameters() if "backbone" not in n]},
         {"params": [p for n, p in policy.named_parameters() if "backbone" in n], "lr": cfg.optimizer_lr_backbone}],
        lr=args.lr, weight_decay=cfg.optimizer_weight_decay)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=True)
    vdl = DataLoader(vds, batch_size=args.batch, num_workers=2)

    def prep(b):
        b["action"] = relative_action_chunk(b["observation.ee_pose"], b["action"])
        return pre(b)

    def save(tag):
        d = out / tag
        policy.save_pretrained(d)
        pre.save_pretrained(d)
        post.save_pretrained(d)

    def evaluate():
        policy.eval()
        l1 = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in vdl:
                b = prep(b)
                b["observation.images"] = [b[k] for k in cfg.image_features]
                pred, _ = policy.model(b)  # eval mode: zero latent, as at deployment
                l1.append(float((F.l1_loss(b["action"], pred, reduction="none")
                                 * ~b["action_is_pad"].unsqueeze(-1)).mean()))
        policy.train()
        return float(np.mean(l1))

    step, best, log = 0, float("inf"), []
    while step < args.steps:
        for b in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, i = policy(prep(b))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0:
                v = evaluate()
                rec = {"step": step, "train_l1": float(i["l1_loss"]), "val_l1": v}
                if v < best:
                    best = v
                    save("best")
                    rec["best"] = True
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step % args.save_every == 0:
                save(f"step_{step:06d}")
            if step >= args.steps:
                break
    save("last")
    (out / "log.json").write_text(json.dumps(log))
    print("best val l1", best)


if __name__ == "__main__":
    main()
