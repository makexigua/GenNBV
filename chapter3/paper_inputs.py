"""Opt-in input adapters. Coordinate bounds retain the published simulator prior.
Normalized continuous actions and inherited world bounds are implementation choices.
"""
import numpy as np
import torch
from gym import spaces


def configure(wrapper):
    cfg = wrapper._gym_env.cfg
    wrapper.continuous_actions = getattr(cfg, 'continuous_actions', False)
    wrapper.grid_channels = getattr(cfg, 'grid_channels', 1)
    if wrapper.continuous_actions:
        wrapper.action_space = spaces.Box(-1., 1., shape=(5,), dtype=np.float32)
    if wrapper.grid_channels == 4:
        old = wrapper.observation_space
        wrapper.observation_space = spaces.Box(
            np.concatenate([old.low[:600], np.full(32000, -np.inf), old.low[8600:]]).astype(np.float32),
            np.concatenate([old.high[:600], np.full(32000, np.inf), old.high[8600:]]).astype(np.float32))


def transform_observation(wrapper, obs):
    if wrapper.grid_channels != 4:
        return obs
    env = wrapper._gym_env
    result = dict(obs)
    # Only coordinates are exposed; ground-truth occupancy is never concatenated.
    result['grid'] = torch.cat([env.grid_world_coordinates, obs['grid'].unsqueeze(-1)], dim=-1).permute(0,4,1,2,3)
    return result


def transform_action(wrapper, action):
    if not wrapper.continuous_actions:
        return action
    env = wrapper._gym_env
    a = torch.as_tensor(action, device=env.device, dtype=torch.float32)
    if a.ndim != 2 or a.shape != (env.num_envs, 5) or not torch.isfinite(a).all():
        raise ValueError('Expected finite batched continuous actions [num_envs,5]')
    indices = [0,1,2,4,5]  # x,y,z,pitch,yaw; roll is fixed.
    low = env.clip_pose_idx_low[indices].float()
    high = env.clip_pose_idx_up[indices].float()
    mapped = low + (a.clamp(-1,1)+1)*0.5*(high-low)
    result = torch.zeros((env.num_envs,6),device=env.device)
    result[:,indices] = mapped  # No rounding: existing affine decoder yields continuous poses.
    return result
