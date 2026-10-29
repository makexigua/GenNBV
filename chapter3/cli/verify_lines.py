"""Server verification of line feature sensitivity, masking, gradients and optional detector."""
import argparse


def make_encoder(use_line, mode='raw'):
    import gym
    import numpy as np
    from gennbv.network.hybrid_encoder import Hybrid_Encoder
    from chapter3.perception.hawp import line_fields
    return Hybrid_Encoder(
        gym.spaces.Box(-np.inf, np.inf, shape=(16792 + (64 * line_fields(mode) if use_line else 0),), dtype=np.float32),
        encoder_param={'hidden_shapes': [256, 256], 'visual_dim': 256},
        net_param={'append_hidden_shapes': [256, 256]},
        state_input_shape=(600,), visual_input_shape=(2, 64, 64),
        use_image=True, use_line=use_line, line_max_lines=64, line_mode=mode).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--hawp-config')
    parser.add_argument('--hawp-weights')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from isaacgym import gymapi  # Must precede torch.
    import numpy as np
    import torch
    from PIL import Image
    from chapter3.perception.hawp import HAWPState
    torch.manual_seed(0)
    model = make_encoder(True)
    obs = torch.rand(2, 16792 + 384)
    obs[:, 16792:] = 0
    changed = obs.clone()
    changed[:, 16792:16798] = torch.tensor([.1, .2, .8, .9, .9, 1.])
    delta = (model(obs) - model(changed)).abs().max().item()
    assert delta > 1e-7, 'Line input has no effect'
    noisy_padding = changed.clone()
    noisy_padding[:, 16792:].view(2, 64, 6)[:, 1:, :5] = 99
    assert torch.allclose(model(changed), model(noisy_padding)), 'Padding leaks into state'
    permuted = changed.clone()
    permuted[:, 16792:] = changed[:, 16792:].view(2, 64, 6).flip(1).reshape(2, -1)
    assert torch.allclose(model(changed), model(permuted), atol=1e-6), 'Line order changes state'
    assert torch.equal(model.encode_lines(torch.zeros(2, 64, 6)), torch.zeros(2, 128))
    model(changed).sum().backward()
    grad = sum(p.grad.abs().sum().item() for p in model.line_encoder.parameters() if p.grad is not None)
    assert grad > 0
    print('PASS: line sensitivity, padding, empty set, order invariance, gradient; delta={}, grad={}'.format(delta, grad))
    evidence_model = make_encoder(True, 'evidence')
    control_model = make_encoder(True, 'unknown')
    control_model.load_state_dict(evidence_model.state_dict(), strict=True)
    line = torch.zeros(2, 64, 9)
    line[:, 0, :6] = torch.tensor([.1, .2, .8, .9, .9, 1.])
    line[:, 0, 8] = 1
    supported = line.clone()
    supported[:, 0, 6] = 1
    supported[:, 0, 8] = 0
    assert not torch.allclose(evidence_model.encode_lines(line), evidence_model.encode_lines(supported))
    assert torch.allclose(evidence_model.encode_lines(line), control_model.encode_lines(line))
    padded = line.clone()
    padded[:, 1:, :5] = 99
    padded[:, 1:, 6:] = 99
    assert torch.allclose(evidence_model.encode_lines(line), evidence_model.encode_lines(padded))
    assert torch.equal(evidence_model.encode_lines(torch.zeros_like(line)), torch.zeros(2, 128))
    evidence_model.encode_lines(supported).sum().backward()
    assert evidence_model.line_encoder[0].weight.grad[:, -3:].abs().sum() > 0
    print('PASS: evidence changes features, evidence gradients, masked padding, identical control architecture')
    if args.image:
        if not args.hawp_config or not args.hawp_weights:
            parser.error('--image requires HAWP config and weights')
        with Image.open(args.image) as image:
            rgb = torch.from_numpy(np.array(image.convert('RGB')))[None].to(args.device)
        detector = HAWPState(args.hawp_config, args.hawp_weights, args.device)
        tokens = detector(rgb)
        assert tokens.shape == (1, 64, 6) and torch.isfinite(tokens).all()
        assert not any(p.requires_grad for p in detector.model.parameters())
        count = int(tokens[..., 5].sum().item())
        assert count > 0, 'No lines detected in this house image; inspect preprocessing/threshold before training'
        print('PASS: official frozen HAWPv2 inference; valid lines={}; {}'.format(count, detector.stats()))


if __name__ == '__main__':
    main()
