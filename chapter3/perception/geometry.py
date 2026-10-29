"""Current-frame depth evidence, not calibrated probabilities or 3D line mapping.

Only aligned positive axial depth and camera intrinsics enter this module.
No scene mesh, segmentation IDs, GT coverage, future frames or rewards are read.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class EvidenceConfig:
    samples: int = 5
    patch_pixels_at_512: float = 6.0
    fit_relative_tolerance: float = 0.02
    jump_relative_scale: float = 0.03
    angle_scale_degrees: float = 15.0
    min_depth: float = 0.001
    max_depth: float = 50.0

    def __post_init__(self):
        values = list(asdict(self).values())
        if (not np.isfinite(values).all() or min(values) <= 0
                or self.samples < 2 or int(self.samples) != self.samples
                or self.min_depth >= self.max_depth):
            raise ValueError('Invalid geometry evidence configuration')


def positive_gym_depth(depth, min_depth=0.001, max_depth=50.0):
    """Gym raw depth is negative axial distance; preserve missing as NaN."""
    z = -np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    return np.where(valid, z, np.nan)


def _plane_at(depth, pixels, center, intrinsics, cfg):
    """Fit 1/z = a*x_ray+b*y_ray+c; a planar slope is not an edge.

    Two trimming passes resist isolated outliers. Fits with poor support or
    residuals return None, so lack of evidence never becomes a negative label.
    """
    h, w = depth.shape
    ij = np.rint(pixels).astype(np.int64)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    ij = np.unique(ij[inside], axis=0)
    if len(ij) < 6:
        return None
    z = depth[ij[:, 1], ij[:, 0]]
    valid = np.isfinite(z) & (z > cfg.min_depth) & (z < cfg.max_depth)
    ij, z = ij[valid], z[valid]
    if len(z) < 6:
        return None
    fx, fy, cx, cy = intrinsics
    design = np.column_stack(((ij[:, 0] - cx) / fx, (ij[:, 1] - cy) / fy, np.ones(len(ij))))
    inv = 1.0 / z
    keep = np.ones(len(z), dtype=bool)
    for _ in range(3):
        if keep.sum() < 6 or np.linalg.cond(design[keep]) > 1e7:
            return None
        coef, _, rank, _ = np.linalg.lstsq(design[keep], inv[keep], rcond=None)
        if rank != 3:
            return None
        residual = np.abs(design @ coef - inv) / inv
        new_keep = residual <= cfg.fit_relative_tolerance
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    # Refit the final subset, then measure its quality.
    if keep.sum() < 6 or np.linalg.cond(design[keep]) > 1e7:
        return None
    coef, _, rank, _ = np.linalg.lstsq(design[keep], inv[keep], rcond=None)
    if rank != 3:
        return None
    relative = (design[keep] @ coef - inv[keep]) / inv[keep]
    rms = float(np.sqrt(np.mean(relative ** 2)))
    quality = (keep.sum() / len(pixels)) * max(0.0, 1 - rms / cfg.fit_relative_tolerance)
    ray = np.array([(center[0] - cx) / fx, (center[1] - cy) / fy, 1.0])
    predicted_inv = float(ray @ coef)
    if predicted_inv <= 0 or not np.isfinite(predicted_inv):
        return None
    predicted_z = 1.0 / predicted_inv
    if not cfg.min_depth < predicted_z < cfg.max_depth:
        return None
    normal = coef / np.linalg.norm(coef)
    return predicted_z, normal, quality


def add_evidence(tokens, depth, intrinsics, config=None):
    """[K,6] -> [K,9]: append support, weak, unknown; keep valid at index 5.

    Their sum is one for a real line and zero for padding. Quality is a
    heuristic fit reliability, NOT the probability that the line is real.
    Both sides must be supported; sky boundaries are retained as unknown.
    """
    cfg = config or EvidenceConfig()
    tokens = np.asarray(tokens, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[1] != 6 or depth.ndim != 2:
        raise ValueError('Expected [K,6] line tokens and aligned [H,W] positive depth')
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or min(intrinsics[:2]) <= 0:
        raise ValueError('Intrinsics must be finite fx,fy,cx,cy with positive focal lengths')
    if not np.isfinite(tokens).all() or np.any((tokens < 0) | (tokens > 1)):
        raise ValueError('Tokens must be finite and normalized')
    result = np.zeros((len(tokens), 9), dtype=np.float32)
    result[:, :6] = tokens
    h, w = depth.shape
    radius = max(3.0, cfg.patch_pixels_at_512 * min(h, w) / 512.0)
    for index, token in enumerate(tokens):
        if token[5] <= 0.5:
            result[index] = 0
            continue
        endpoints = token[:4].reshape(2, 2) * [w, h]
        delta = endpoints[1] - endpoints[0]
        length = np.linalg.norm(delta)
        result[index, 8] = 1.0
        if length < 2 * radius:
            continue
        tangent = delta / length
        perpendicular = np.array([-tangent[1], tangent[0]])
        masses = []
        for t in np.linspace(0.15, 0.85, cfg.samples):
            center = endpoints[0] + t * delta
            sides = []
            for sign in (-1, 1):
                pixels = np.array([center + along * tangent + sign * across * perpendicular
                                   for along in (-radius, 0, radius)
                                   for across in (radius / 3, 2 * radius / 3, radius)])
                sides.append(_plane_at(depth, pixels, center, intrinsics, cfg))
            if any(side is None for side in sides):
                masses.append([0., 0., 1.])
                continue
            z1, n1, q1 = sides[0]
            z2, n2, q2 = sides[1]
            jump = abs(z1 - z2) / max(min(z1, z2), cfg.min_depth)
            angle = np.degrees(np.arccos(np.clip(abs(n1 @ n2), 0., 1.)))
            support = np.clip(max(jump / cfg.jump_relative_scale,
                                  angle / cfg.angle_scale_degrees), 0., 1.)
            quality = min(q1, q2)
            masses.append([quality * support, quality * (1 - support), 1 - quality])
        result[index, 6:] = np.mean(masses, axis=0)
    return result


def unknown_evidence(tokens):
    """Same-size control with no depth information; valid lines are unknown."""
    result = np.zeros((*tokens.shape[:-1], 9), dtype=np.float32)
    result[..., :6] = tokens
    valid = tokens[..., 5] > .5
    result[~valid] = 0
    result[..., 8] = valid
    return result
