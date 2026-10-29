"""Server entry point for existing train_gennbv options, including --use_line."""
import argparse
import functools
import os
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sim_device', default='cuda:0')
    known, _ = parser.parse_known_args()
    if not known.sim_device.startswith('cuda:'):
        raise ValueError('This runner requires an explicit cuda:N device')
    gpu = int(known.sim_device.split(':')[1])
    os.environ['CUDA_DEVICE'] = str(gpu)
    from isaacgym import gymapi
    import torch
    torch.cuda.set_device(gpu)
    import gennbv.utils as utils
    utils.SourceModule = functools.partial(utils.SourceModule, arch='sm_86')
    from gennbv.train.train_gennbv import main as train
    train()
    print('TRAINING_COMPLETE', flush=True)
    sys.stderr.flush()
    os._exit(0)  # Only after train returned and the environment closed successfully.


if __name__ == '__main__':
    main()
