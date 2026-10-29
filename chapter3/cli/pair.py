"""Train a HAWP state policy, then evaluate it and an existing matched image baseline.

Use a tmux session. This runs real training; --help does not import Isaac Gym.
"""
import argparse
import json
from pathlib import Path
import sys


from chapter3.cli._process import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path,
                        help='Three-channel final checkpoint, trained with the same seed and 262144 steps')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--hawp-config', required=True, type=Path)
    parser.add_argument('--hawp-weights', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--graphics', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from chapter3.paths import REPO_ROOT
    root = REPO_ROOT
    if Path.cwd().resolve() != root:
        parser.error('Run from the upstream directory')
    # Current training BaseTask uses the simulation ordinal for graphics.
    if args.gpu != args.graphics:
        parser.error('Training currently requires matching CUDA/Vulkan ordinals; use verified 0/0')
    for path in (args.baseline, args.hawp_config, args.hawp_weights):
        if not path.is_file():
            parser.error('Missing {}'.format(path))
    compare = root.parents[1] / '_artifacts/compare_gennbv_eval.py'
    if not compare.is_file():
        parser.error('Sync the thesis comparison script first')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'experiment.json').write_text(json.dumps(vars(args), default=str, indent=2))
    python = sys.executable
    model, complete = run([
        python, '-u', '-m', 'chapter3', 'train', '--sim_device', 'cuda:{}'.format(args.gpu),
        '--rl_device', 'cuda:{}'.format(args.gpu), '--graphics_device_id', str(args.graphics),
        '--pipeline', 'gpu', '--headless', '--stop_wandb', '--num_envs', '256',
        '--seed', str(args.seed), '--n_steps', '128', '--batch_size', '128',
        '--total_iters', '8', '--n_epochs', '5', '--max_episode_length', '100',
        '--use_image', '1', '--use_line', '1', '--hawp_config', str(args.hawp_config),
        '--hawp_weights', str(args.hawp_weights), '--exp_name', 'pilot256_hawp_s{}'.format(args.seed)
    ], args.output / 'train.log')
    if not complete or model is None or not model.is_file():
        raise RuntimeError('Training did not produce a verified final checkpoint')
    for label, weights in [('baseline', args.baseline), ('hawp', model)]:
        command = [python, '-u', '-m', 'chapter3', 'eval', '--weights', str(weights),
                   '--device', str(args.gpu), '--graphics-device', str(args.graphics),
                   '--num-envs', '50', '--views', '30', '--seed', '100', '--use-image',
                   '--output', str(args.output / label)]
        if label == 'hawp':
            command += ['--use-line', '--hawp-config', str(args.hawp_config),
                        '--hawp-weights', str(args.hawp_weights)]
        run(command, args.output / (label + '_eval.log'))
        if not json.loads((args.output / label / 'summary.json').read_text()).get('completed'):
            raise RuntimeError('Incomplete evaluation: ' + label)
    run([python, str(compare), str(args.output / 'baseline'), str(args.output / 'hawp'),
         '--comparison', 'hawp'], args.output / 'comparison.txt')
    print('HAWP_COMPARISON_COMPLETE: {}'.format(args.output / 'comparison.txt'))


if __name__ == '__main__':
    main()
