"""Run with exporter Python; semantic checks independent of a trained network."""

import torch
from umi_policy import normalize_pose, predict, prepare_observation


def main():
    pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05]).repeat(
        1, 2, 1
    )
    images = torch.zeros((1, 2, 3, 48, 64))
    normalized = normalize_pose(pose, 1.0, 0.1)
    assert torch.equal(normalized[..., :9], pose[..., :9])
    assert torch.equal(normalized[..., 9], torch.zeros((1, 2)))
    batch = prepare_observation(pose, images, 1.0, 0.1)
    assert batch["observation.images"].shape == (1, 2, 1, 3, 192, 256)
    assert torch.allclose(
        batch["observation.images"][0, 0, 0, :, 0, 0],
        torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]),
    )
    bad = pose.clone()
    bad[..., 0] = 1.1
    try:
        normalize_pose(bad, 1.0, 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range labels were accepted")

    class FakeModel:
        def eval(self):
            return self

        def _prepare_global_conditioning(self, batch):
            return batch["observation.state"]

        def conditional_sample(self, batch_size, global_cond):
            result = normalized[:, :1].repeat(batch_size, 16, 1)
            result[0, :, 0] = torch.arange(1, 17) / 100
            return result

    action = predict(FakeModel(), pose, images, 1.0, 0.1)
    assert action.shape == (1, 16, 10)
    # No past-action offset may remove target 1, despite two observations.
    assert torch.allclose(action[0, :, 0], torch.arange(1, 17) / 100)
    assert torch.allclose(action[0, :, 9], torch.full((16,), 0.05))
    assert torch.allclose(action[0, :, 3:6].norm(dim=-1), torch.ones(16))
    print(
        "UMI policy normalization, temporal camera layout and future-index checks passed"
    )


if __name__ == "__main__":
    main()
