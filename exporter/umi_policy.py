"""Matching, offline diffusion baseline for yumi.umi.relative_tcp_10d.v1.

Uses the pinned LeRobot diffusion model, but explicitly predicts ONLY future
poses. Its standard generate_actions/select_action offset convention is not used.
No code in this module connects to, or commands, a robot.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from torch.utils.data import DataLoader, Dataset, Subset

CONTRACT = "yumi.umi.relative_tcp_10d.v1"
IMAGE_SIZE = (192, 256)


def normalize_pose(value, translation_scale, aperture_scale):
    out = value.clone()
    out[..., :3] /= translation_scale
    out[..., 9] = 2 * out[..., 9] / aperture_scale - 1
    if not torch.isfinite(out).all() or out.abs().max() > 1.0001:
        raise ValueError(
            "Pose outside normalization range; adjust measured scale, never silently clip labels"
        )
    return out


def prepare_observation(state, images, translation_scale, aperture_scale):
    """B,H,10 and B,H,3,Y,X -> model tensors; images are RGB floats in [0,1]."""
    if (
        state.ndim != 3
        or state.shape[-1] != 10
        or images.ndim != 5
        or images.shape[:2] != state.shape[:2]
        or images.shape[2] != 3
    ):
        raise ValueError("Expected matching temporal pose and RGB histories")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Images must be finite RGB floats in [0,1]")
    b, h = images.shape[:2]
    images = F.interpolate(
        images.flatten(0, 1),
        size=IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return {
        "observation.state": normalize_pose(state, translation_scale, aperture_scale),
        "observation.images": ((images - mean) / std).reshape(b, h, 1, 3, *IMAGE_SIZE),
    }


class Chunks(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.contract = json.loads((self.root / "umi-contract.json").read_text())
        if (
            self.contract["contract"] != CONTRACT
            or not self.contract["calibration_verified"]
        ):
            raise ValueError("A verified, matching UMI chunk dataset is required")
        self.ds = LeRobotDataset(
            "local/single-arm-umi", root=self.root, video_backend="pyav"
        )
        self.synthetic = all(
            p["calibration"].get("tracking_serial") == "SYNTHETIC"
            for p in self.contract["provenance"]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        nominal = torch.arange(1, self.contract["horizon"] + 1) / self.contract["fps"]
        if not torch.allclose(row["action.time_offsets"], nominal, atol=0.012, rtol=0):
            raise ValueError("Action timing exceeds fixed-rate policy tolerance")
        return {
            "state": row["observation.state"],
            "images": torch.stack(
                [row[k] for k in self.contract["history_image_keys"]]
            ),
            "action": row["action"],
        }


def model_config(history, horizon):
    if horizon % 4:
        raise ValueError("This baseline needs horizon divisible by four")
    cfg = DiffusionConfig(
        n_obs_steps=history,
        horizon=horizon,
        n_action_steps=min(8, horizon - history + 1),
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (10,)),
            "observation.images.d405": PolicyFeature(
                FeatureType.VISUAL, (3, *IMAGE_SIZE)
            ),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (10,))},
        crop_shape=None,
        pretrained_backbone_weights=None,
        down_dims=(64, 128, 256),
        diffusion_step_embed_dim=64,
        noise_scheduler_type="DDIM",
        num_inference_steps=10,
        device="cpu",
    )
    cfg.validate_features()
    return cfg


def batch_tensors(row, device, translation_scale, aperture_scale):
    batch = prepare_observation(
        row["state"].to(device),
        row["images"].to(device),
        translation_scale,
        aperture_scale,
    )
    batch["action"] = normalize_pose(
        row["action"].to(device), translation_scale, aperture_scale
    )
    batch["action_is_pad"] = torch.zeros(
        batch["action"].shape[:2], dtype=torch.bool, device=device
    )
    return batch


@torch.inference_mode()
def predict(model, state, images, translation_scale, aperture_scale):
    model.eval()
    batch = prepare_observation(state, images, translation_scale, aperture_scale)
    # LeRobot's standard generate_actions skips n_obs_steps-1 predictions.
    # Our dataset stores only future actions, so sample the full future horizon.
    condition = model._prepare_global_conditioning(batch)
    action = model.conditional_sample(len(state), global_cond=condition)
    action[..., :3] *= translation_scale
    action[..., 9] = (action[..., 9] + 1) * aperture_scale / 2
    # Project predicted 6D rotations onto SO(3), rejecting degenerate predictions.
    x = action[..., 3:6]
    y = action[..., 6:9]
    nx = x.norm(dim=-1, keepdim=True)
    if torch.any(nx < 1e-6):
        raise ValueError("Degenerate predicted rotation")
    x = x / nx
    y = y - (x * y).sum(dim=-1, keepdim=True) * x
    ny = y.norm(dim=-1, keepdim=True)
    if torch.any(ny < 1e-6):
        raise ValueError("Degenerate predicted rotation")
    action[..., 3:6] = x
    action[..., 6:9] = y / ny
    if not torch.isfinite(action).all():
        raise ValueError("Nonfinite prediction")
    return action


def train(args):
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.translation_scale <= 0
        or args.aperture_scale <= 0
    ):
        raise ValueError("Positive steps, batch size and scales required")
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    data = Chunks(args.dataset)
    if args.smoke_test != data.synthetic:
        raise ValueError(
            "Use --smoke-test only with explicitly synthetic data; real training requires real data"
        )
    history, horizon = data.contract["history"], data.contract["horizon"]
    # Split whole SOURCE episodes, not neighboring overlapping action chunks.
    mapping = data.contract["source_mapping"]
    groups = sorted({(m["source_root"], m["source_episode"]) for m in mapping})
    if not args.smoke_test and len(groups) < 2:
        raise ValueError(
            "Need at least two source episodes for disjoint training/validation"
        )
    validation = {groups[-1]} if len(groups) > 1 else set()
    val_episodes = {
        m["output_episode"]
        for m in mapping
        if (m["source_root"], m["source_episode"]) in validation
    }
    episode_ids = data.ds.hf_dataset["episode_index"]
    train_indices = [i for i, e in enumerate(episode_ids) if int(e) not in val_episodes]
    val_indices = [i for i, e in enumerate(episode_ids) if int(e) in val_episodes]
    loader = DataLoader(
        Subset(data, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(Subset(data, val_indices), batch_size=args.batch_size, num_workers=0)
        if val_indices
        else None
    )
    model = DiffusionModel(model_config(history, horizon)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    iterator = iter(loader)
    losses = []
    root.mkdir(parents=True)
    for step in range(args.steps):
        try:
            row = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            row = next(iterator)
        batch = batch_tensors(
            row, args.device, args.translation_scale, args.aperture_scale
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == args.steps - 1:
            print(
                f"step {step + 1}/{args.steps}: diffusion loss {losses[-1]:.6f}",
                flush=True,
            )
    val_loss = None
    if val_loader:
        model.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for row in val_loader:
                batch = batch_tensors(
                    row, args.device, args.translation_scale, args.aperture_scale
                )
                n = len(row["state"])
                total += float(model.compute_loss(batch)) * n
                count += n
        val_loss = total / count
    payload = {
        "contract": CONTRACT,
        "history": history,
        "horizon": horizon,
        "fps": data.contract["fps"],
        "translation_scale": args.translation_scale,
        "aperture_scale": args.aperture_scale,
        "synthetic": data.synthetic,
        "hardware_validated": False,
        "steps": args.steps,
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, root / "policy.pt")
    # Reload the saved artifact and exercise the inference path, not just forward loss.
    restored = torch.load(root / "policy.pt", map_location="cpu", weights_only=True)
    check = DiffusionModel(model_config(history, horizon))
    check.load_state_dict(restored["model"])
    row = data[train_indices[0]]
    prediction = predict(
        check,
        row["state"][None],
        row["images"][None],
        args.translation_scale,
        args.aperture_scale,
    )
    assert prediction.shape == (1, horizon, 10)
    report = {
        "contract": CONTRACT,
        "synthetic": data.synthetic,
        "steps": args.steps,
        "last_training_loss": losses[-1],
        "validation_loss": val_loss,
        "training_chunks": len(train_indices),
        "validation_chunks": len(val_indices),
        "held_out_source_episodes": [list(g) for g in sorted(validation)],
        "checkpoint_reload_and_prediction_passed": True,
        "hardware_validated": False,
        "source_dataset": str(data.root.resolve()),
        "dataset_contract": data.contract,
    }
    (root / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "dataset_contract"}, indent=2
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--translation-scale", type=float, default=1.0)
    p.add_argument("--aperture-scale", type=float, default=0.1)
    p.add_argument("--smoke-test", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
