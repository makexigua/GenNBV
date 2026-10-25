"""Inference-only smoke test for the supplied GenNBV c373f76 source.

Place this file in the upstream repository root. No training is performed.
Results are diagnostic coverage traces, not a reproduction of paper metrics.
"""
import argparse
import csv
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--device', type=int, default=0, help='CUDA device index')
    p.add_argument('--graphics-device', type=int, required=True,
                   help='Vulkan device index of the same physical GPU')
    p.add_argument('--num-envs', type=int, default=1, choices=range(1, 51))
    p.add_argument('--views', type=int, default=5, help='Counted env.step calls; reset is warm-up')
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--output', default='_artifacts/pretrained_smoke')
    p.add_argument('--check-only', action='store_true', help='Check policy weights without creating simulator')
    p.add_argument('--stochastic', action='store_true')
    args = p.parse_args()
    if not 2 <= args.views <= 100:
        p.error('--views must be between 2 and 100')
    return args


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    if not (root / 'gennbv').is_dir():
        raise RuntimeError('Place eval_pretrained.py in GenNBV/upstream/')
    os.chdir(root)
    sys.path.insert(0, str(root))
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    # pycuda.autoinit uses this choice when gennbv is imported.
    os.environ['CUDA_DEVICE'] = str(args.device)
    from isaacgym import gymapi  # Must precede torch.
    import torch
    import numpy as np
    from gym.spaces import Box, MultiDiscrete
    from PIL import Image
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    import gennbv
    import gennbv.utils as nbv_utils
    from gennbv.env.env_eval_gennbv import Env_Eval_GenNBV
    from gennbv.env.config_gennbv_eval import Config_GenNBV_Eval
    from gennbv.network.hybrid_encoder import Hybrid_Encoder
    from gennbv.wrapper.env_wrapper_gennbv_eval import EnvWrapperGenNBVEval
    from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
    from stable_baselines3.common.policies import ActorCriticPolicy_Train_Eval

    set_seed(args.seed)
    # CUDA 11.7 cannot compile sm_89, which PyCUDA auto-detects on a 4090.
    # Ampere sm_86 code is binary compatible with Ada sm_89.
    nbv_utils.SourceModule = functools.partial(nbv_utils.SourceModule, arch='sm_86')

    # Mirror the architecture in train_eval_gennbv.py, without PPO rollout buffers.
    # Its encoder mutates append_hidden_shapes via pop(), so reconstruct fresh kwargs.
    obs_space = Box(-np.inf, np.inf, shape=(600 + 8000 + 2 * 64 * 64,), dtype=np.float32)
    action_space = MultiDiscrete([81, 81, 51, 1, 13, 13])
    policy = ActorCriticPolicy_Train_Eval(
        obs_space, action_space, lr_schedule=lambda _: 0.0,
        net_arch=[], ortho_init=False, features_extractor_class=Hybrid_Encoder,
        features_extractor_kwargs=dict(
            encoder_param={'hidden_shapes': [256, 256], 'visual_dim': 256},
            net_param={'transformer_params': [[1, 256], [1, 256]],
                       'append_hidden_shapes': [256]},
            state_input_shape=(600,), visual_input_shape=(100, 400, 400),
        ),
    )
    # Official SB3 checkpoint: load the policy only, not optimizer or pickled metadata.
    with ZipFile(weights) as archive:
        if 'policy.pth' not in archive.namelist():
            raise RuntimeError('Expected an SB3 checkpoint containing policy.pth')
        with archive.open('policy.pth') as f:
            state = torch.load(f, map_location='cpu')
    current = policy.state_dict()
    problems = {
        'missing': sorted(set(current) - set(state)),
        'unexpected': sorted(set(state) - set(current)),
        'shape_mismatch': {
            k: [list(state[k].shape), list(current[k].shape)]
            for k in set(state) & set(current) if state[k].shape != current[k].shape
        },
    }
    if any(problems.values()):
        raise RuntimeError('Checkpoint/source architecture mismatch; no partial loading:\n'
                           + json.dumps(problems, indent=2))
    policy.load_state_dict(state, strict=True)
    del state
    policy.to(device).set_training_mode(False)
    print('Strict checkpoint loading passed.', flush=True)
    print('Encoder in this source uses pose history and occupancy, not RGB.', flush=True)
    if args.check_only:
        return

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous run.
    for i in range(args.num_envs):
        for rel in (f'data_gennbv/eval/urdf/house_batch12_setA_{i+1}.urdf',
                    f'data_gennbv/eval/obj/houses3k/BAT12_SETA_HOUSE{i+1}.obj',
                    f'data_gennbv/eval/gt/point_cloud/BAT12_SETA_HOUSE{i+1}_pc.pt'):
            if not (root / rel).is_file():
                raise FileNotFoundError(root / rel)
    if not (root / 'data_gennbv/eval/gt/eval_houses3k_grid_gt.pt').is_file():
        raise FileNotFoundError('Missing eval_houses3k_grid_gt.pt')

    class RecordedEval(Env_Eval_GenNBV):
        def create_sim(self):
            self.graphics_device_id = args.graphics_device
            super().create_sim()

        def render(self, sync_frame_time=True):
            # Upstream queries viewer keyboard events even when viewer is None.
            # Sensor rendering still occurs in get_step_return().
            if self.viewer is not None:
                return super().render(sync_frame_time)

        def post_process_camera_tensor(self):
            super().post_process_camera_tensor()
            # Called while access to camera tensors is held by the original env.
            self.record_rgb = torch.stack(self.rgb_cam_tensors)[..., :3].cpu().numpy().copy()
            self.record_depth = self.depth_processed.cpu().numpy().copy()

        def _reward_surface_coverage(self):
            reward = super()._reward_surface_coverage()
            # Capture before upstream reset_idx clears terminal coverage and point clouds.
            self.record_coverage = (self.scanned_gt_grid.sum((1, 2, 3)) /
                                    self.num_valid_voxel_gt).detach().cpu().numpy().copy()
            self.record_poses = self.poses.detach().cpu().numpy().copy()
            self.record_actions = self.actions.detach().cpu().numpy().copy()
            return reward

    cfg = Config_GenNBV_Eval()
    cfg.env.num_envs = args.num_envs
    cfg.max_episode_length = args.views
    cfg.visual_input.stack = 100  # Must match the checkpoint, independent of view budget.
    sim_args = SimpleNamespace(physics_engine=gymapi.SIM_PHYSX, use_gpu=True,
                               use_gpu_pipeline=True, subscenes=0, num_threads=2)
    sim_params = parse_sim_params(sim_args, {'sim': class_to_dict(cfg.sim)})
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    meta = vars(args).copy()
    meta.update(weights=str(weights), checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                metric='diagnostic voxel coverage; not paper benchmark AUC',
                reset_observation_counted=False, scene_ids=list(range(1, args.num_envs + 1)),
                source_encoder='pose history + occupancy; RGB unused',
                pycuda_compile_arch='sm_86', torch=torch.__version__)
    meta['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    raw = None
    completed = False
    records = []
    finished = np.zeros(args.num_envs, dtype=bool)
    try:
        raw = RecordedEval(cfg, sim_params, gymapi.SIM_PHYSX, str(device), True)
        wrapped = EnvWrapperGenNBVEval(raw)
        if wrapped.observation_space.shape != obs_space.shape:
            raise RuntimeError('Observation shape differs from checkpoint architecture')
        if not np.array_equal(wrapped.action_space.nvec, action_space.nvec):
            raise RuntimeError('Action space differs from checkpoint architecture')
        obs, rewards, dones, infos, accuracies = wrapped.reset()
        with (out / 'trace.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['scene_id', 'view', 'coverage', 'done',
                                                 'collision', 'pose', 'action'])
            writer.writeheader()
            for view in range(1, args.views + 1):
                # Upstream reset() returns a reset sentinel done=1 and clears buffers.
                # Like its evaluation loop, discard reset rewards/dones and count steps.
                # The first step is forced to the configured initial pose by upstream.
                with torch.no_grad():
                    actions, _ = policy.predict(obs, deterministic=not args.stochastic)
                obs, rewards, dones, infos, accuracies = wrapped.step(actions)
                done_np = dones.detach().cpu().numpy().astype(bool)
                collision = raw.collision_buf.detach().cpu().numpy().astype(bool)
                for i in range(args.num_envs):
                    if finished[i]:
                        continue
                    row = dict(scene_id=i + 1, view=view, coverage=float(raw.record_coverage[i]),
                               done=bool(done_np[i]), collision=bool(collision[i]),
                               pose=json.dumps(raw.record_poses[i].tolist()),
                               action=json.dumps(raw.record_actions[i].tolist()))
                    writer.writerow(row)
                    records.append(row)
                    scene_dir = out / f'house_{i+1:02d}'
                    scene_dir.mkdir(exist_ok=True)
                    Image.fromarray(raw.record_rgb[i].astype(np.uint8)).save(scene_dir / f'rgb_{view:03d}.png')
                    np.save(scene_dir / f'depth_processed_{view:03d}.npy', raw.record_depth[i])
                    finished[i] = done_np[i]
                    print(f'house={i+1} view={view} coverage={row["coverage"]:.4f} done={done_np[i]}', flush=True)
                f.flush()
                if finished.all():
                    break
        last = {row['scene_id']: row for row in records}
        summary = {
            'completed': True, 'scene_count': len(last),
            'mean_final_voxel_coverage': float(np.mean([r['coverage'] for r in last.values()])),
            'scenes': list(last.values()),
            'note': 'First counted episode only; reset warm-up excluded, first step is fixed initial pose; collisions may terminate early. '
                    'No paper AUC or centimetre accuracy claim. Depth is upstream-processed, not raw.',
        }
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for scene in sorted(last):
            rows = [r for r in records if r['scene_id'] == scene]
            ax.plot([r['view'] for r in rows], [100*r['coverage'] for r in rows],
                    marker='o' if len(last) <= 10 else None,
                    linewidth=1 if len(last) > 10 else 1.5,
                    alpha=0.5 if len(last) > 10 else 1,
                    label=f'House {scene}' if len(last) <= 10 else None)
        ax.set(xlabel='Counted view (reset warm-up excluded)', ylabel='Voxel coverage (%)',
               ylim=(0, 100), title='Diagnostic run, not paper benchmark')
        if len(last) <= 10:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'coverage.png', dpi=150)
        plt.close(fig)
        print('Saved diagnostic results to', out, flush=True)
        completed = True
    finally:
        if raw is not None:
            raw.close()
        if completed:
            # Isaac Gym/PyCUDA can segfault after close() during Python teardown.
            # All result files and the simulator are closed before this point.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == '__main__':
    main()
