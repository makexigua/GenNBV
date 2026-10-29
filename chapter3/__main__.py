"""One entry point for Chapter 3 workflows. Heavy imports are deferred."""
import argparse
import importlib
import sys

COMMANDS = {
    'train': ('chapter3.cli.train', 'Train with existing GenNBV arguments'),
    'eval': ('chapter3.cli.evaluate', 'Evaluate a saved policy'),
    'ablate': ('chapter3.cli.ablate', 'Run four matched groups (smoke by default)'),
    'pair': ('chapter3.cli.pair', 'Earlier HAWP versus saved image-baseline workflow'),
    'inspect': ('chapter3.cli.inspect', 'Audit saved geometry evidence'),
    'verify-lines': ('chapter3.cli.verify_lines', 'Verify line encoder and optional HAWP inference'),
    'verify-image': ('chapter3.cli.verify_image', 'Verify image encoder'),
    'detect-legacy': ('chapter3.perception.legacy_lines', 'Earlier LSD/DeepLSD visualization tool'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Subcommand options: python -m chapter3 COMMAND --help')
    parser.add_argument('command', choices=list(COMMANDS), help='; '.join(k + ': ' + v[1] for k, v in COMMANDS.items()))
    # Parse only the command; preserve --help for the selected command.
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = ['python -m chapter3 ' + args.command] + sys.argv[2:]
    importlib.import_module(COMMANDS[args.command][0]).main()


if __name__ == '__main__':
    main()
