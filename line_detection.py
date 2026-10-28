"""Standalone RGB line detection; no Isaac Gym imports or learned confidence."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def detect_lines(rgb, min_length=20.0, max_lines=64):
    """Return pixel endpoints sorted by length. Length is not confidence."""
    if min_length <= 0 or max_lines <= 0:
        raise ValueError('min_length and max_lines must be positive')
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB image [H, W, 3]')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    lines = lines.reshape(-1, 4)
    lengths = np.linalg.norm(lines[:, 2:] - lines[:, :2], axis=1)
    order = np.argsort(-lengths, kind='stable')
    return lines[order[lengths[order] >= min_length][:max_lines]].copy()


def save_detection(rgb, prefix, min_length=20.0, max_lines=64):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    targets = [Path(str(prefix) + suffix) for suffix in ('.json', '.png')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Detection output already exists: {}'.format(prefix))
    lines = detect_lines(rgb, min_length, max_lines)
    overlay = Image.fromarray(rgb)
    draw = ImageDraw.Draw(overlay)
    for x1, y1, x2, y2 in lines:
        draw.line((float(x1), float(y1), float(x2), float(y2)), fill=(0, 255, 0), width=2)
    payload = dict(detector='OpenCV LSD_REFINE_STD', opencv_version=cv2.__version__,
                   width=rgb.shape[1], height=rgb.shape[0],
                   coordinate_system='pixels; origin top-left; x right, y down',
                   min_length_pixels=min_length, max_lines=max_lines,
                   lines_xyxy=lines.tolist(),
                   note='2D detections only; no geometry validation or confidence score.')
    targets[0].write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    overlay.save(targets[1])
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='One RGB image or evaluation directory')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--min-length', type=float, default=20.0)
    parser.add_argument('--max-lines', type=int, default=64)
    args = parser.parse_args()
    sources = [args.input] if args.input.is_file() else sorted(args.input.rglob('rgb_*.png'))
    if not sources:
        parser.error('No RGB images found')
    args.output.mkdir(parents=True, exist_ok=False)
    for source in sources:
        relative = Path(source.name) if args.input.is_file() else source.relative_to(args.input)
        with Image.open(source) as im:
            rgb = np.asarray(im.convert('RGB'))
        count = save_detection(rgb, args.output / relative.with_suffix(''), args.min_length, args.max_lines)
        print('{}: {} lines'.format(relative, count))


if __name__ == '__main__':
    main()
