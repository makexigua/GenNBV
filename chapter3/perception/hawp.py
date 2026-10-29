"""Frozen official HAWPv2 observations. Detector scores are NOT geometric confidence."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from dataclasses import asdict
from chapter3.perception.geometry import EvidenceConfig, add_evidence, unknown_evidence


def line_fields(mode):
    if mode not in ('raw', 'evidence', 'unknown'):
        raise ValueError('Unknown line mode: ' + str(mode))
    return 6 if mode == 'raw' else 9


def configure_line_camera(env, detector):
    env.capture_line_rgb = detector is not None
    env.capture_line_seg = detector is not None
    env.capture_line_depth = detector is not None and detector.geometry_mode == 'evidence'
    if not env.capture_line_depth:
        return
    cfg = env.cfg.visual_input
    cameras = []
    for scene, handle in zip(env.envs, env.camera_handles):
        projection = np.asarray(env.gym.get_camera_proj_matrix(env.sim, scene, handle))
        cameras.append([abs(projection[0, 0]) * cfg.camera_width / 2,
                        abs(projection[1, 1]) * cfg.camera_height / 2,
                        cfg.camera_width / 2, cfg.camera_height / 2])
    cameras = np.asarray(cameras, dtype=np.float64)
    if cameras.shape != (env.num_envs, 4) or not np.isfinite(cameras).all() or np.any(cameras[:, :2] <= 0):
        raise ValueError('Invalid sensor projection matrices')
    if not np.allclose(cameras, cameras[0]):
        raise ValueError('This experiment expects identical camera intrinsics across environments')
    env.line_intrinsics = cameras
    detector.manifest['camera'] = dict(width=cfg.camera_width, height=cfg.camera_height,
                                      intrinsics=cameras[0].tolist(),
                                      near=float(cfg.near_plane),
                                      depth_limit=float(min(abs(env.DEPTH_SENSE_DIST), cfg.far_plane)),
                                      source='Gym raw axial depth, symmetric pinhole projection')

LINE_FIELDS = 6  # x1/W, y1/H, x2/W, y2/H, detector score, valid mask


class _NoLines(Exception):
    """Official HAWPv2 can reach an empty proposal set before reshape/min."""


def foreground_line_mask(lines, foreground, samples=16, radius=4, minimum_fraction=0.25):
    """Keep lines with enough samples touching the target or its silhouette."""
    foreground = np.asarray(foreground, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError('Foreground mask must be [H,W]')
    h, w = foreground.shape
    keep = np.zeros(len(lines), dtype=bool)
    for i, line in enumerate(lines):
        endpoints = line.reshape(2, 2)
        points = endpoints[0] + np.linspace(0.05, 0.95, samples)[:, None] * (endpoints[1] - endpoints[0])
        hits = 0
        for x, y in points:
            ix, iy = int(round(x)), int(round(y))
            x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
            y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
            hits += x0 < x1 and y0 < y1 and foreground[y0:y1, x0:x1].any()
        keep[i] = hits / samples >= minimum_fraction
    return keep


def pack_lines(lines, scores, width, height, max_lines=64, threshold=0.5, foreground=None):
    if max_lines < 1 or width < 1 or height < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid line packing settings')
    lines = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(lines) != len(scores):
        raise ValueError('Line and score counts differ')
    keep = np.isfinite(lines).all(1) & np.isfinite(scores) & (scores >= threshold)
    lines, scores = lines[keep], scores[keep]
    if foreground is not None:
        if np.asarray(foreground).shape != (height, width):
            raise ValueError('Foreground and RGB shapes differ')
        keep = foreground_line_mask(lines, foreground)
        lines, scores = lines[keep], scores[keep]
    lines = np.clip(lines / np.array([width, height, width, height]), 0, 1)
    # Canonical endpoint ordering makes undirected segments identical after reversal.
    reverse = (lines[:, 0] > lines[:, 2]) | ((lines[:, 0] == lines[:, 2]) & (lines[:, 1] > lines[:, 3]))
    lines[reverse] = lines[reverse][:, [2, 3, 0, 1]]
    keep = np.linalg.norm(lines[:, 2:] - lines[:, :2], axis=1) > 1e-6
    lines, scores = lines[keep], scores[keep]
    order = np.argsort(-scores, kind='stable')[:max_lines]
    result = np.zeros((max_lines, LINE_FIELDS), dtype=np.float32)
    result[:len(order), :4] = lines[order]
    result[:len(order), 4] = np.clip(scores[order], 0, 1)
    result[:len(order), 5] = 1
    return result


class HAWPState:
    def __init__(self, config, weights, device, max_lines=64, threshold=0.5,
                 geometry_mode='raw', geometry_config=None):
        import torch
        self.geometry_mode = geometry_mode
        self.fields = line_fields(geometry_mode)
        self.geometry_config = geometry_config or EvidenceConfig()
        if not Path(config).is_file() or not Path(weights).is_file():
            raise FileNotFoundError('Provide official hawpv2.yaml and hawpv2-edb9b23f.pth')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Official HAWPv2 uses CUDA internally; select cuda:N')
        if max_lines < 1 or not 0 <= threshold <= 1:
            raise ValueError('Invalid line settings')
        self.max_lines, self.threshold = max_lines, threshold
        # Official constructor calls .cuda(); keep it on the selected physical GPU.
        # Loading the frozen detector must not change the policy RNG stream.
        with torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.device.index or 0]):
            from hawp.fsl.config import cfg
            from hawp.fsl.dataset.build import build_transform
            from hawp.fsl.model.build import build_model
            config_obj = cfg.clone()
            config_obj.merge_from_file(str(config))
            config_obj.MODEL.DEVICE = str(self.device)
            self.model = build_model(config_obj).to(self.device)
            checkpoint = torch.load(str(weights), map_location='cpu')
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.model.eval().requires_grad_(False)
            self.transform = build_transform(config_obj)
        original_matcher = self.model.wireframe_matcher

        def guarded_matcher(junctions, proposals, *args, **kwargs):
            if len(junctions) < 2 or len(proposals) == 0:
                raise _NoLines()
            result = original_matcher(junctions, proposals, *args, **kwargs)
            if len(result[0]) == 0:
                raise _NoLines()
            return result

        self.model.wireframe_matcher = guarded_matcher
        self.manifest = {
            'detector': 'HAWPv2', 'frozen': True, 'schema': 'xyxy_score_mask_v1',
            'max_lines': max_lines, 'threshold': threshold,
            'weights_sha256': hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            'config_sha256': hashlib.sha256(Path(config).read_bytes()).hexdigest(),
            'input': 'current full-resolution RGB; official resize/normalization',
            'foreground_filter': 'target_segmentation_255_touch_4px_4of16_v1',
        }
        if geometry_mode != 'raw':
            self.manifest.update(schema='xyxy_score_mask_support_weak_unknown_v1',
                                 geometry_mode=geometry_mode,
                                 geometry_config=asdict(self.geometry_config),
                                 geometry_algorithm='local_inverse_depth_planes_v1')
        import hawp
        repo = Path(hawp.__file__).resolve().parent.parent
        try:
            self.manifest['hawp_commit'] = subprocess.check_output(
                ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            self.manifest['hawp_commit'] = 'unknown'
        self.seconds = 0.0
        self.images = 0
        self.valid_lines = 0
        self.geometry_seconds = 0.0
        self.evidence_sum = np.zeros(3, dtype=np.float64)

    def __call__(self, rgb, depth=None, intrinsics=None, foreground=None):
        import torch
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError('HAWP expects uint8 [N,H,W,3] RGB captured this step')
        if foreground is None or tuple(foreground.shape) != tuple(rgb.shape[:3]):
            raise ValueError('HAWP state requires aligned current-frame target mask')
        start = time.perf_counter()
        images = rgb.detach().cpu().numpy()
        masks = foreground.detach().cpu().numpy()
        tokens = []
        with torch.no_grad(), torch.cuda.device(self.device):
            for image, mask in zip(images, masks):
                height, width = image.shape[:2]
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                try:
                    output, _ = self.model(tensor, [{'height': height, 'width': width, 'filename': 'online'}])
                except _NoLines:
                    tokens.append(np.zeros((self.max_lines, LINE_FIELDS), dtype=np.float32))
                    continue
                tokens.append(pack_lines(output['lines_pred'].cpu().numpy(),
                                         output['lines_score'].cpu().numpy(), width, height,
                                         self.max_lines, self.threshold, foreground=mask))
        array = np.stack(tokens)
        self.seconds += time.perf_counter() - start  # includes transfers and official CPU preprocessing
        self.images += len(images)
        self.valid_lines += int(array[..., 5].sum())
        geometry_start = time.perf_counter()
        if self.geometry_mode == 'evidence':
            if depth is None or intrinsics is None or tuple(depth.shape) != tuple(rgb.shape[:3]):
                raise ValueError('Geometry requires same-frame raw axial depth aligned with RGB and intrinsics')
            depth_np = depth.detach().cpu().numpy()
            if np.asarray(intrinsics).shape != (len(images), 4):
                raise ValueError('Expected per-camera fx,fy,cx,cy')
            array = np.stack([add_evidence(token, z, camera, self.geometry_config)
                              for token, z, camera in zip(array, depth_np, intrinsics)])
        elif self.geometry_mode == 'unknown':
            array = unknown_evidence(array)
        if self.geometry_mode != 'raw':
            self.geometry_seconds += time.perf_counter() - geometry_start
            self.evidence_sum += array[..., 6:].sum(axis=(0, 1))
        return torch.from_numpy(array).to(rgb.device)

    def stats(self):
        return dict(images=self.images, seconds=self.seconds,
                    ms_per_image=1000 * self.seconds / max(1, self.images),
                    mean_valid_lines=self.valid_lines / max(1, self.images),
                    geometry_seconds=self.geometry_seconds,
                    geometry_ms_per_image=1000 * self.geometry_seconds / max(1, self.images),
                    mean_evidence_masses=(self.evidence_sum / max(1, self.valid_lines)).tolist())


def training_arguments():
    return [
        dict(name='--use_line', type=int, default=0),
        dict(name='--hawp_config', type=str, default=''),
        dict(name='--hawp_weights', type=str, default=''),
        dict(name='--line_max_lines', type=int, default=64),
        dict(name='--line_threshold', type=float, default=0.5),
        dict(name='--line_mode', type=str, default='raw', choices=['raw', 'evidence', 'unknown']),
    ]
