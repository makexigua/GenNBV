"""Compatibility entry; prefer: python -m chapter3 eval ..."""
from chapter3.cli.evaluate import main, parse_args

if __name__ == "__main__":
    main()
