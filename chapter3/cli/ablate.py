"""Train/evaluate matched Chapter 3 ablations sequentially. Default: tiny smoke test.

Run in upstream under the activated environment with the working graphics setup.
All groups train from scratch; GPU availability must be checked before launch.
"""
import argparse
import json
from pathlib import Path
import sys
from chapter3.cli._process import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['smoke', 'pilot'], default='smoke')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--graphics', type=int, default=0)
    p.add_argument('--hawp-config', type=Path, required=True)
    p.add_argument('--hawp-weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from chapter3.paths import REPO_ROOT
    root = REPO_ROOT
    if Path.cwd().resolve() != root or args.gpu != args.graphics:
        p.error('Run from upstream; use matching CUDA/Vulkan indices (verified 0/0)')
    compare = root.parents[1] / '_artifacts/compare_gennbv_eval.py'
    for path in (args.hawp_config, args.hawp_weights, compare):
        if not path.is_file():
            p.error('Missing ' + str(path))
    args.output.mkdir(parents=True, exist_ok=False)
    settings = dict(envs=2, steps=8, iterations=1, epochs=1, batch=8, episode=2, eval_envs=1, views=2)
    if args.stage == 'pilot':
        settings.update(envs=256, steps=128, iterations=8, epochs=5, batch=128,
                        episode=100, eval_envs=50, views=30)
    experiment = dict(arguments=vars(args), settings=settings, models={},
                      status='running', note='Pilot, not converged results or paper benchmark')
    experiment_path = args.output / 'experiment.json'

    def save():
        experiment_path.write_text(json.dumps(experiment, default=str, indent=2))

    save()
    try:
        for mode in ('baseline', 'raw', 'unknown', 'evidence'):
            command = [sys.executable, '-u', '-m', 'chapter3', 'train',
                       '--sim_device', 'cuda:{}'.format(args.gpu), '--rl_device', 'cuda:{}'.format(args.gpu),
                       '--graphics_device_id', str(args.graphics), '--pipeline', 'gpu',
                       '--headless', '--stop_wandb', '--seed', str(args.seed), '--use_image', '1',
                       '--num_envs', str(settings['envs']), '--n_steps', str(settings['steps']),
                       '--total_iters', str(settings['iterations']), '--n_epochs', str(settings['epochs']),
                       '--batch_size', str(settings['batch']), '--max_episode_length', str(settings['episode']),
                       '--lr', '0.0001', '--exp_name', 'ch3_{}_{}_s{}'.format(args.stage, mode, args.seed)]
            if mode != 'baseline':
                command += ['--use_line', '1', '--line_mode', mode,
                            '--hawp_config', str(args.hawp_config), '--hawp_weights', str(args.hawp_weights)]
            model, completed = run(command, args.output / (mode + '_train.log'))
            if not completed or model is None or not model.is_file():
                raise RuntimeError('Training did not complete: ' + mode)
            experiment['models'][mode] = str(model)
            save()
            command = [sys.executable, '-u', '-m', 'chapter3', 'eval', '--weights', str(model),
                       '--device', str(args.gpu), '--graphics-device', str(args.graphics),
                       '--use-image', '--seed', '100', '--num-envs', str(settings['eval_envs']),
                       '--views', str(settings['views']), '--output', str(args.output / mode)]
            if mode != 'baseline':
                command += ['--use-line', '--line-mode', mode, '--hawp-config', str(args.hawp_config),
                            '--hawp-weights', str(args.hawp_weights)]
            run(command, args.output / (mode + '_eval.log'))
            summary = json.loads((args.output / mode / 'summary.json').read_text())
            if not summary.get('completed'):
                raise RuntimeError('Incomplete evaluation: ' + mode)
        run([sys.executable, '-m', 'chapter3', 'inspect', str(args.output / 'evidence')],
            args.output / 'evidence_audit.txt')
        for a, b, comparison in [('baseline', 'raw', 'hawp'), ('raw', 'evidence', 'geometry'),
                                 ('unknown', 'evidence', 'geometry')]:
            run([sys.executable, str(compare), str(args.output / a), str(args.output / b),
                 '--comparison', comparison], args.output / (a + '_vs_' + b + '.txt'))
    except BaseException:
        experiment['status'] = 'failed_or_interrupted'
        save()
        raise
    experiment['status'] = 'completed'
    save()
    print('CHAPTER3_{}_COMPLETE: {}'.format(args.stage.upper(), args.output))


if __name__ == '__main__':
    main()
