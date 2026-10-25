"""Check that the optional image branch reads pixels and receives gradients."""

from isaacgym import gymapi  # Import before torch, as required by Isaac Gym.
import gym
import numpy as np
import torch

from gennbv.network.hybrid_encoder import Hybrid_Encoder


def make_encoder(use_image):
    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(600 + 8000 + 2 * 64 * 64,), dtype=np.float32
    )
    return Hybrid_Encoder(
        obs_space,
        encoder_param={"hidden_shapes": [256, 256], "visual_dim": 256},
        net_param={"transformer_params": [[1, 256], [1, 256]],
                   "append_hidden_shapes": [256, 256]},
        state_input_shape=(600,),
        visual_input_shape=(2, 64, 64),
        use_image=use_image,
    )


def main():
    torch.manual_seed(0)
    original = torch.zeros(2, 600 + 8000 + 2 * 64 * 64)
    changed = original.clone()
    changed[:, 8600:] = 255.0

    legacy = make_encoder(False).eval()
    with torch.no_grad():
        assert torch.equal(legacy(original), legacy(changed))

    image_model = make_encoder(True).eval()
    with torch.no_grad():
        difference = (image_model(original) - image_model(changed)).abs().max().item()
    assert difference > 1e-6, "Changing image pixels did not change the features"

    image_model.train()
    image_model(changed).sum().backward()
    image_grad = sum(
        p.grad.abs().sum().item() for p in image_model.naive_encoder_image.parameters()
        if p.grad is not None
    )
    assert image_grad > 0, "Image encoder did not receive a gradient"
    print(f"PASS: image changes output (max delta={difference:.6g}); "
          f"image branch has gradient (L1={image_grad:.6g})")
    print("PASS: legacy branch ignores images, preserving pretrained behavior")


if __name__ == "__main__":
    main()
