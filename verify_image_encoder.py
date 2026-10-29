"""Compatibility entry; prefer: python -m chapter3 verify-image"""
from chapter3.cli.verify_image import main, make_encoder

if __name__ == "__main__":
    main()
