"""CPU architecture checks inside the server's GenNBV Python environment."""
def main():
    from isaacgym import gymapi  # Must precede torch on the server.
    import io
    import gym
    import numpy as np
    import torch
    from torch import nn
    from gennbv.network.hybrid_encoder import Hybrid_Encoder
    from stable_baselines3.common.policies import ActorCriticPolicy_Train_Eval
    obs = gym.spaces.Box(-np.inf, np.inf, shape=(16792,), dtype=np.float32)
    action = gym.spaces.MultiDiscrete([81, 81, 51, 1, 13, 13])
    for convs, hidden, channels, continuous in ((3, 0, 1, False), (2, 2, 4, True)):
        size=600+8000*channels+8192
        obs=gym.spaces.Box(-np.inf,np.inf,shape=(size,),dtype=np.float32)
        action=gym.spaces.Box(-1.,1.,shape=(5,),dtype=np.float32) if continuous else gym.spaces.MultiDiscrete([81,81,51,1,13,13])
        def build():
            return ActorCriticPolicy_Train_Eval(
                obs, action, lr_schedule=lambda _: 1e-4,
                net_arch=[dict(pi=[256,256], vf=[256,256])] if hidden else [],
                features_extractor_class=Hybrid_Encoder,
                features_extractor_kwargs=dict(encoder_param={},
                    net_param={'append_hidden_shapes':[256]},
                    state_input_shape=(600,), visual_input_shape=(2,64,64),
                    use_image=True, image_conv_layers=convs, grid_channels=channels))
        policy=build().eval()
        assert sum(isinstance(m,nn.Conv2d) for m in policy.features_extractor.modules())==convs
        assert sum(isinstance(m,nn.Linear) for m in policy.mlp_extractor.policy_net.modules())==hidden
        x=torch.randn(2,size)
        features=policy.extract_features(x)
        assert features.shape==(2,256)
        features.square().mean().backward()
        assert policy.features_extractor.naive_encoder_image[0].weight.grad.abs().sum()>0
        with torch.no_grad():
            expected=policy(x,deterministic=True)
        buffer=io.BytesIO()
        torch.save(policy.state_dict(),buffer); buffer.seek(0)
        restored=build().eval(); restored.load_state_dict(torch.load(buffer),strict=True)
        with torch.no_grad():
            actual=restored(x,deterministic=True)
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b)
        print('PASS conv_layers={} policy_hidden_layers={} shape/gradient/roundtrip'.format(convs,hidden))
    from types import SimpleNamespace
    from chapter3.paper_inputs import transform_action, transform_observation
    env=SimpleNamespace(device='cpu',num_envs=2,
        clip_pose_idx_low=torch.zeros(6),clip_pose_idx_up=torch.tensor([80,80,50,0,12,12]))
    wrapper=SimpleNamespace(_gym_env=env,continuous_actions=True,grid_channels=4)
    actions=transform_action(wrapper,torch.tensor([[-1.,-1.,-1.,-1.,-1.],[1.,1.,1.,1.,1.]]))
    torch.testing.assert_close(actions[0],torch.zeros(6))
    torch.testing.assert_close(actions[1],env.clip_pose_idx_up.float())
    assert transform_action(wrapper,torch.full((2,5),0.123))[0,0].item() != round(transform_action(wrapper,torch.full((2,5),0.123))[0,0].item())
    env.grid_world_coordinates=torch.randn(2,20,20,20,3)
    occupancy=torch.zeros(2,20,20,20)
    result=transform_observation(wrapper,{'grid':occupancy})['grid']
    assert result.shape==(2,4,20,20,20)
    torch.testing.assert_close(result[:,3],occupancy)
    torch.testing.assert_close(result[:,:3].permute(0,2,3,4,1),env.grid_world_coordinates)
    print('PASS continuous bounds/fractional action and coordinate/occupancy separation')
if __name__=='__main__': main()
