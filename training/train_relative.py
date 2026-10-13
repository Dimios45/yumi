"""ACT+CLIP and Diffusion Policy (n_obs_steps=2) on vruga/yumi-umi-block-place, UMI relative actions.

Same relative representation as train_act_relative.py: observation.state is the dataset's UMI-relative
proprioception; action chunks are converted to poses relative to the *current* EE pose (the last
observation step) before normalisation.

  python train_relative.py --policy act_clip
  python train_relative.py --policy diffusion
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "pydeps"))  # transformers (CLIP), kept out of the shared env

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features

from train_act_relative import relative_action_chunk, relative_to_absolute  # noqa: F401  (re-exported)

FPS = 30
IMG = "observation.images.wrist"
OBS_STRIDE = 3  # frames between observation steps = dataset obs_step_frames (0.1 s, UMI-style 10 Hz)


class ClipBackbone(nn.Module):
    """CLIP ViT patch tokens as an ACT feature map. Input is dataset-normalised, any resolution."""

    def __init__(self, img_stats, name="openai/clip-vit-base-patch16", hw=(224, 304)):
        super().__init__()
        from transformers import CLIPVisionModel
        self.vit = CLIPVisionModel.from_pretrained(name)
        self.hw, self.patch = hw, self.vit.config.patch_size
        self.out_dim = self.vit.config.hidden_size
        for k, v in (("ds_mean", img_stats["mean"]), ("ds_std", img_stats["std"])):
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x = F.interpolate(x, size=self.hw, mode="bilinear", antialias=True, align_corners=False)
        x = ((x * self.ds_std + self.ds_mean) - self.clip_mean) / self.clip_std
        tok = self.vit(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        h, w = self.hw[0] // self.patch, self.hw[1] // self.patch
        return {"feature_map": tok.transpose(1, 2).reshape(x.shape[0], self.out_dim, h, w)}


def build_act_clip(cfg, img_stats):
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy(cfg)
    bb = ClipBackbone(img_stats)
    policy.model.backbone = bb
    policy.model.encoder_img_feat_input_proj = nn.Conv2d(bb.out_dim, cfg.dim_model, kernel_size=1)
    return policy


def relative_stats(root, horizon, start):
    """Stats of relative action chunks whose first element is at offset `start` frames from t."""
    df = pd.read_parquet(Path(root) / "data" / "chunk-000" / "file-000.parquet")
    chunks, poses = [], []
    for _, ep in df.groupby("episode_index"):
        act = torch.tensor(np.stack(ep["action"].values), dtype=torch.float32)
        ee = torch.tensor(np.stack(ep["observation.ee_pose"].values), dtype=torch.float32)
        idx = (torch.arange(len(ep))[:, None] + start + torch.arange(horizon)[None]).clamp(0, len(ep) - 1)
        chunks.append(act[idx])
        poses.append(ee)
    rel = relative_action_chunk(torch.cat(poses), torch.cat(chunks)).reshape(-1, 10)
    return {"mean": rel.mean(0), "std": rel.std(0).clamp_min(1e-6), "min": rel.min(0).values, "max": rel.max(0).values}


def resize_images(batch, hw):
    x = batch[IMG]
    lead = x.shape[:-3]
    batch[IMG] = F.interpolate(x.reshape(-1, *x.shape[-3:]), size=hw, mode="bilinear", antialias=True,
                               align_corners=False).reshape(*lead, 3, *hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["act_clip", "diffusion"], required=True)
    ap.add_argument("--root", default="/home/sra/vrushtee/yumi_act/data")
    ap.add_argument("--repo-id", default="vruga/yumi-umi-block-place")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--resume", type=Path, default=None, help="Checkpoint directory to load weights from")
    ap.add_argument("--start-step", type=int, default=0, help="Completed steps in the checkpoint")
    args = ap.parse_args()
    dev = "cuda"
    diff = args.policy == "diffusion"
    steps = args.steps or (40000 if diff else 20000)
    if not 0 <= args.start_step < steps:
        ap.error("--start-step must be nonnegative and smaller than --steps")
    if args.start_step and args.resume is None:
        ap.error("--start-step requires --resume")
    if args.resume and not (args.resume / "model.safetensors").is_file():
        ap.error("--resume must contain model.safetensors")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this trainer in a GPU-enabled session")
    batch_size = args.batch or (64 if diff else 32)
    lr = args.lr or 1e-4
    out = Path(args.out or f"/home/sra/vrushtee/yumi_act/outputs/{args.policy}_relative")
    out.mkdir(parents=True, exist_ok=True)

    # ---- dataset timing -------------------------------------------------------------------------
    if diff:
        n_obs, horizon, n_act = 2, 32, 16
        obs_ts = [-OBS_STRIDE / FPS, 0.0]
        act_start = 1 - n_obs  # lerobot Diffusion: chunk index 0 sits one step before the last obs
        delta = {k: obs_ts for k in (IMG, "observation.state", "observation.ee_pose")}
        delta["action"] = [(act_start + i) / FPS for i in range(horizon)]
        hw = (240, 320)
    else:
        n_obs, horizon, n_act, act_start, hw = 1, 50, 50, 0, None
        delta = {"action": [i / FPS for i in range(horizon)]}
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav", delta_timestamps=delta)
    feats = dataset_to_policy_features(ds.meta.features)

    from lerobot.configs.types import PolicyFeature
    in_feats = {k: f for k, f in feats.items() if k in (IMG, "observation.state")}
    if hw:
        in_feats[IMG] = PolicyFeature(type=in_feats[IMG].type, shape=(3, *hw))
    common = dict(input_features=in_feats, output_features={"action": feats["action"]}, device=dev)

    # ---- policy ---------------------------------------------------------------------------------
    stats = {k: v for k, v in ds.meta.stats.items() if k in (IMG, "observation.state")}
    stats["action"] = relative_stats(args.root, horizon, act_start)
    stats = {k: {s: torch.as_tensor(x) for s, x in v.items()} for k, v in stats.items()}
    if diff:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
        cfg = DiffusionConfig(n_obs_steps=n_obs, horizon=horizon, n_action_steps=n_act,
                              crop_shape=(216, 288), crop_is_random=True,
                              pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1", use_group_norm=False,
                              do_mask_loss_for_padding=True, optimizer_lr=lr, **common)
        policy = DiffusionPolicy(cfg)
        pre, post = make_diffusion_pre_post_processors(cfg, dataset_stats=stats)
        params = policy.parameters()
        opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=cfg.optimizer_betas, eps=cfg.optimizer_eps,
                                weight_decay=cfg.optimizer_weight_decay)
        warm = cfg.scheduler_warmup_steps
        for group in opt.param_groups:
            group["initial_lr"] = lr
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)),
            last_epoch=args.start_step - 1)
    else:
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
        cfg = ACTConfig(chunk_size=horizon, n_action_steps=n_act, pretrained_backbone_weights=None,
                        optimizer_lr=lr, **common)
        policy = build_act_clip(cfg, stats[IMG])
        pre, post = make_act_pre_post_processors(cfg, dataset_stats=stats)
        opt = torch.optim.AdamW(
            [{"params": [p for n, p in policy.named_parameters() if "backbone" not in n]},
             {"params": [p for n, p in policy.named_parameters() if "backbone" in n], "lr": cfg.optimizer_lr_backbone}],
            lr=lr, weight_decay=cfg.optimizer_weight_decay)
        sched = None
    if args.resume:
        from safetensors.torch import load_model
        load_model(policy, str(args.resume / "model.safetensors"))
        print(f"Loaded weights from {args.resume}; continuing at step {args.start_step} "
              "with a fresh optimizer", flush=True)
    policy.to(dev).train()
    print(f"{args.policy}: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params, "
          f"bs={batch_size}, steps={steps}", flush=True)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=True)

    def save(tag):
        d = out / tag
        policy.save_pretrained(d)
        pre.save_pretrained(d)
        post.save_pretrained(d)

    step, log = args.start_step, []
    while step < steps:
        for batch in dl:
            cur = batch["observation.ee_pose"]
            cur = cur[:, -1] if diff else cur  # current pose = last observation step
            batch["action"] = relative_action_chunk(cur, batch["action"])
            if hw:
                resize_images(batch, hw)
            batch = pre(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, info = policy(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            if sched:
                sched.step()
            step += 1
            if step % 100 == 0:
                rec = {"step": step, "loss": loss.item(), **{k: float(v) for k, v in (info or {}).items()}}
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step % args.save_every == 0:
                save(f"step_{step:06d}")
            if step >= steps:
                break
    save("last")
    (out / "log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
