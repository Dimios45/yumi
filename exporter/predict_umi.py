"""Predict one offline UMI chunk from a checkpoint; emits no hardware commands."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from umi_policy import CONTRACT, Chunks, model_config, predict


def run(checkpoint, dataset, index, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    data = Chunks(dataset)
    if payload["contract"] != CONTRACT or any(
        payload[k] != data.contract[k] for k in ("history", "horizon", "fps")
    ):
        raise ValueError("Checkpoint/data contract mismatch")
    model = DiffusionModel(model_config(payload["history"], payload["horizon"]))
    model.load_state_dict(payload["model"])
    row = data[index]
    action = predict(
        model,
        row["state"][None],
        row["images"][None],
        payload["translation_scale"],
        payload["aperture_scale"],
    )[0]
    doc = {
        "contract": CONTRACT,
        "calibration_verified": data.contract["calibration_verified"],
        "synthetic_checkpoint": payload["synthetic"],
        "hardware_execution_allowed": False,
        "source_dataset": str(Path(dataset).resolve()),
        "source_sample_index": index,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "samples": [
            {
                "observation_state": row["state"].tolist(),
                "action": action.tolist(),
                "action_time_offsets_s": [
                    (i + 1) / payload["fps"] for i in range(payload["horizon"])
                ],
            }
        ],
        "note": "Policy prediction, not a recorded demonstration. Offline inspection/IK only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"Predicted {len(action)} future targets: {output}; no hardware commands sent"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.checkpoint, a.dataset, a.index, a.output)
