"""Recompute evidence from saved online sensor snapshots before long training."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    args = parser.parse_args()
    import numpy as np
    from chapter3.perception.geometry import EvidenceConfig, add_evidence
    meta = json.loads((args.evaluation / 'metadata.json').read_text())
    if meta.get('line_mode') != 'evidence':
        parser.error('Requires an evaluation saved with --line-mode evidence')
    manifest = meta['hawp']
    cfg = EvidenceConfig(**manifest['geometry_config'])
    camera = manifest['camera']
    all_masses = []
    frames = 0
    depth_count = depth_total = 0
    for path in sorted(args.evaluation.glob('house_*/line_tokens_*.npy')):
        view = path.stem.rsplit('_', 1)[-1]
        depth = np.load(path.with_name('line_depth_m_' + view + '.npy'), allow_pickle=False)
        tokens = np.load(path, allow_pickle=False)
        if depth.shape != (camera['height'], camera['width']):
            raise ValueError('Sensor shape differs from metadata: ' + str(path))
        recalculated = add_evidence(tokens[:, :6], depth, camera['intrinsics'], cfg)
        np.testing.assert_allclose(tokens, recalculated, atol=1e-5, rtol=1e-5)
        valid = tokens[:, 5] > .5
        np.testing.assert_allclose(tokens[valid, 6:].sum(-1), 1., atol=1e-5)
        all_masses.extend(tokens[valid, 6:].tolist())
        frames += 1
        depth_count += int(np.isfinite(depth).sum())
        depth_total += depth.size
    if not frames:
        parser.error('No saved evidence frames')
    mean = np.mean(all_masses, axis=0).tolist() if all_masses else None
    print(json.dumps(dict(snapshot_recomputation_passed=True, frames=frames, lines=len(all_masses),
                          valid_depth_fraction=depth_count / depth_total,
                          mean_support_weak_unknown=mean,
                          note='Inspect overlays too; this checks reproducibility, not semantic correctness.'), indent=2))


if __name__ == '__main__':
    main()
