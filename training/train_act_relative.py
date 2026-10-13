"""Train LeRobot ACT on vruga/yumi-umi-block-place with UMI relative-trajectory actions.

* observation.state  : already UMI-relative in the dataset (pose 3 frames ago relative to the
                       current pose + prev/current jaw width)               -> PD2.2
* action             : stored absolute in the episode-initial frame; every chunk is converted
                       here to poses relative to the *current* EE pose        -> PD2.1
                       (rel pos, rel rot6d, absolute width)

Policy output is therefore a relative chunk; use `relative_to_absolute` (or compose with the
live EE pose) at deployment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

FPS = 30


def rot6d_to_matrix(d6):
    a, b = d6[..., :3], d6[..., 3:6]
    x = a / a.norm(dim=-1, keepdim=True)
    b = b - (x * b).sum(-1, keepdim=True) * x
    y = b / b.norm(dim=-1, keepdim=True)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)


def matrix_to_rot6d(R):
    return R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def pose_to_T(p):
    T = torch.zeros(*p.shape[:-1], 4, 4, dtype=p.dtype, device=p.device)
    T[..., :3, :3] = rot6d_to_matrix(p[..., 3:9])
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1
    return T


def relative_action_chunk(current_pose, chunk):
    """(...,10), (...,H,10) -> (...,H,10): chunk poses relative to current pose, width absolute."""
    rel = torch.linalg.inv(pose_to_T(current_pose.double()))[..., None, :, :] @ pose_to_T(chunk.double())
    return torch.cat([rel[..., :3, 3], matrix_to_rot6d(rel[..., :3, :3]), chunk[..., 9:10].double()], -1).float()


def relative_to_absolute(current_pose, rel_chunk):
    """Inverse of relative_action_chunk (for deployment): compose with the live EE pose."""
    ab = pose_to_T(current_pose.double())[..., None, :, :] @ pose_to_T(rel_chunk.double())
    return torch.cat([ab[..., :3, 3], matrix_to_rot6d(ab[..., :3, :3]), rel_chunk[..., 9:10].double()], -1).float()


def relative_action_stats(root, horizon):
    """Mean/std/min/max of the relative action chunks over the whole dataset (no video decode)."""
    df = pd.read_parquet(Path(root) / "data" / "chunk-000" / "file-000.parquet")
    chunks, poses = [], []
    for _, ep in df.groupby("episode_index"):
        act = torch.tensor(np.stack(ep["action"].values), dtype=torch.float32)
        ee = torch.tensor(np.stack(ep["observation.ee_pose"].values), dtype=torch.float32)
        idx = (torch.arange(len(ep))[:, None] + torch.arange(horizon)[None]).clamp(max=len(ep) - 1)
        chunks.append(act[idx])
        poses.append(ee)
    rel = relative_action_chunk(torch.cat(poses), torch.cat(chunks)).reshape(-1, 10)
    return {
        "mean": rel.mean(0), "std": rel.std(0).clamp_min(1e-6),
        "min": rel.min(0).values, "max": rel.max(0).values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/sra/vrushtee/yumi_act/data")
    ap.add_argument("--repo-id", default="vruga/yumi-umi-block-place")
    ap.add_argument("--out", default="/home/sra/vrushtee/yumi_act/outputs/act_relative")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=5000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    ts = [i / FPS for i in range(args.chunk)]
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav",
                        delta_timestamps={"action": ts})
    feats = dataset_to_policy_features(ds.meta.features)
    cfg = ACTConfig(
        input_features={k: f for k, f in feats.items() if k in ("observation.images.wrist", "observation.state")},
        output_features={"action": feats["action"]},
        chunk_size=args.chunk, n_action_steps=args.chunk,
        optimizer_lr=args.lr, device=dev,
    )

    stats = {k: v for k, v in ds.meta.stats.items() if k in ("observation.images.wrist", "observation.state")}
    stats["action"] = relative_action_stats(args.root, args.chunk)
    stats = {k: {s: torch.as_tensor(x) for s, x in v.items()} for k, v in stats.items()}

    policy = ACTPolicy(cfg).to(dev).train()
    pre, post = make_act_pre_post_processors(cfg, dataset_stats=stats)
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad]},
         {"params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad],
          "lr": cfg.optimizer_lr_backbone}],
        lr=args.lr, weight_decay=cfg.optimizer_weight_decay)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)

    def save(tag):
        d = out / tag
        policy.save_pretrained(d)
        pre.save_pretrained(d)
        post.save_pretrained(d)

    step, log = 0, []
    while step < args.steps:
        for batch in dl:
            # PD2.1: chunk relative to the current EE pose, before normalization
            batch["action"] = relative_action_chunk(batch["observation.ee_pose"], batch["action"])
            batch = pre(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, info = policy(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "loss": loss.item(), **{k: float(v) for k, v in info.items()}}
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step % args.save_every == 0:
                save(f"step_{step:06d}")
            if step >= args.steps:
                break
    save("last")
    (out / "log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
