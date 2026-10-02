#!/usr/bin/env python3
"""Estimate the variance shape from noisy SIDD raw crops and export a training YAML.

Run from coding/. Inference reads only datasets.train and resume.path.teacher1.
Alternatively fit an existing cells CSV on a machine without PyTorch.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

CODING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODING_ROOT))
from src.utils.noise_calibration import calibrated_training_config, fit_affine_shape


def coding_path(value):
    path = Path(value)
    return path if path.is_absolute() else CODING_ROOT / path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_crops(data_dir, pattern, max_groups, crops_per_group, seed):
    """Choose one recorded observation per condition, with multiple prepared crops."""
    regex = re.compile(pattern)
    if regex.groups < 1:
        raise ValueError('--group-regex must contain a capture group for condition ID')
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix.lower() != '.mat':
            continue
        match = regex.search(path.name)
        if not match:
            raise ValueError('Cannot identify capture condition: {}'.format(path.name))
        observation = re.sub(r'_s\d+$', '', path.stem, flags=re.IGNORECASE)
        groups[match.group(1)][observation].append(path)
    if not groups:
        raise ValueError('No prepared noisy MAT crops found in {}'.format(data_dir))
    rng = random.Random(seed)
    group_ids = sorted(groups)
    if max_groups and len(group_ids) > max_groups:
        group_ids = sorted(rng.sample(group_ids, max_groups))
    selected = []
    for group in group_ids:
        observation = sorted(groups[group])[0]
        paths = groups[group][observation]
        if len(paths) > crops_per_group:
            paths = sorted(rng.sample(paths, crops_per_group))
        selected.extend((group, path) for path in paths)
    return selected


def collect_cells(config, args):
    # Heavy dependencies are needed only when producing predictions.
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.io import loadmat
    from src.model.backbone.unet import UNet

    train_data = config['datasets']['train']
    if train_data['type'] not in ('DataLoader_SIDD_Medium_Raw', 'DataLoader_SIDD_Medium_Raw_mod2'):
        raise ValueError('Direct inference currently supports prepared SIDD raw noisy MAT crops only')
    teacher_spec = dict(config['models']['sub_models']['teacher1'])
    if teacher_spec.pop('type') != 'UNet':
        raise ValueError('Direct calibration currently supports the raw UNet separator')
    state_key, checkpoint_name = config['resume']['path']['teacher1']
    checkpoint = coding_path(checkpoint_name)
    patch = train_data['patch']
    if patch < 64 or patch % 64:
        raise ValueError('Bayer patch size must be a positive multiple of 64 for this UNet')
    params = config['train']['loss_params']
    ws, threshold = params.get('ws', 7), params['thres']
    if ws < 1 or ws % 2 == 0 or ws // 2 >= patch // 2:
        raise ValueError('Invalid local-standard-deviation window')
    selected = select_crops(coding_path(train_data['data_dir']), args.group_regex,
                            args.max_groups, args.crops_per_group, args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or config['device'])
    teacher = UNet(**teacher_spec)
    state = torch.load(str(checkpoint), map_location='cpu')
    for key in str(state_key).split('.'):
        state = state[key]
    teacher.load_state_dict(state, strict=True)
    teacher.requires_grad_(False)
    teacher.eval().to(device)
    rng = random.Random(args.seed)
    rows, manifest = [], []
    with torch.no_grad():
        for index, (group, path) in enumerate(selected):
            raw = np.asarray(loadmat(str(path))['x'], dtype=np.float32)
            if raw.ndim != 2 or min(raw.shape) < patch or not np.isfinite(raw).all():
                raise ValueError('Expected a finite Bayer array at least patch-sized: {}'.format(path))
            # Preserve Bayer phase and sample horizontal/vertical offsets independently.
            top = 2 * rng.randrange((raw.shape[0] - patch) // 2 + 1)
            left = 2 * rng.randrange((raw.shape[1] - patch) // 2 + 1)
            crop = np.ascontiguousarray(raw[top:top + patch, left:left + patch])
            noisy = torch.from_numpy(crop)[None, None].to(device)
            noisy = F.unfold(noisy, 2, stride=2).view(1, 4, patch // 2, patch // 2)
            prediction = teacher(noisy)
            residual = noisy - prediction
            gray = prediction.mean(dim=1, keepdim=True) * 255.0
            padded = F.pad(gray, [ws // 2] * 4, mode='reflect')
            windows = F.unfold(padded, ws)
            local_std = ((windows - windows.mean(dim=1, keepdim=True)) ** 2).mean(dim=1).sqrt()
            mask = (local_std.view_as(gray) < threshold).to(prediction.dtype)

            # Nonoverlapping 8x8 cells per packed raw channel. The mask is
            # common across channels; each cell's residual mean is removed.
            mask_cells = F.unfold(mask, 8, stride=8).transpose(1, 2)
            counts = mask_cells.sum(dim=-1)
            pred_cells = F.unfold(prediction, 8, stride=8).view(4, 64, -1).transpose(1, 2)
            residual_cells = F.unfold(residual, 8, stride=8).view(4, 64, -1).transpose(1, 2)
            means = (pred_cells * mask_cells).sum(dim=-1) / counts.clamp(min=1)
            residual_means = (residual_cells * mask_cells).sum(dim=-1) / counts.clamp(min=1)
            variances = (((residual_cells - residual_means[..., None]) ** 2) * mask_cells).sum(dim=-1)
            variances = variances / (counts - 1).clamp(min=1)
            valid = ((counts >= args.min_pixels) & torch.isfinite(means)
                     & torch.isfinite(variances) & (means >= 0) & (means <= 1))
            means, variances = means.cpu().tolist(), variances.cpu().tolist()
            counts, valid = counts[0].cpu().tolist(), valid.cpu().tolist()
            for channel in range(4):
                for cell, keep in enumerate(valid[channel]):
                    if keep:
                        rows.append({'group': group, 'mean': means[channel][cell],
                                     'variance': variances[channel][cell], 'count': int(counts[cell]),
                                     'file': path.name, 'channel': channel, 'cell': cell})
            manifest.append({'group': group, 'file': str(path), 'top': top, 'left': left})
            print('Calibration crop {}/{}; retained cells {}'.format(index + 1, len(selected), len(rows)), flush=True)
    return rows, {'source': 'noisy training crops', 'checkpoint': str(checkpoint),
                  'checkpoint_key': state_key, 'checkpoint_sha256': file_hash(checkpoint),
                  'seed': args.seed, 'patch': patch, 'window': ws, 'threshold': threshold,
                  'cell_size': 8, 'group_regex': args.group_regex,
                  'selection': 'one recorded observation per capture condition', 'manifest': manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Existing BSS stdnorm training YAML')
    parser.add_argument('--output-dir', required=True, help='Writes cells.csv, fit.json and train.yaml')
    parser.add_argument('--statistics-csv', help='Fit existing group,mean,variance,count cells without inference')
    parser.add_argument('--device', help='Inference device, e.g. cuda:0 or cpu; defaults to YAML')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--max-groups', type=int, default=32, help='Maximum fitting conditions; 0 selects all')
    parser.add_argument('--crops-per-group', type=int, default=8)
    parser.add_argument('--group-regex', default=r'^(\d{4})_', help='First capture group identifies the condition')
    parser.add_argument('--min-pixels', type=int, default=32)
    args = parser.parse_args(argv)
    if args.max_groups < 0 or args.crops_per_group < 1 or not 2 <= args.min_pixels <= 64:
        parser.error('Require max-groups >= 0, crops-per-group >= 1 and 2 <= min-pixels <= 64')
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if config['models']['main_model']['type'] not in (
            'OneTeacher_TwoBranch_StdNorm', 'OneTeacher_TwoBranch_StdNorm_SIDDRaw'):
        parser.error('Use one of the training YAMLs in configs/bss/stdnorm')
    output = Path(args.output_dir).resolve()
    for name in ('cells.csv', 'fit.json', 'train.yaml'):
        if (output / name).exists():
            parser.error('Output already exists: {}'.format(output / name))
    if args.statistics_csv:
        stats_path = Path(args.statistics_csv).resolve()
        with stats_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if not {'group', 'mean', 'variance', 'count'} <= set(reader.fieldnames or ()):
                parser.error('Statistics CSV requires group,mean,variance,count columns')
            rows = list(reader)
        provenance = {'source': 'supplied local statistics', 'path': str(stats_path),
                      'sha256': file_hash(stats_path)}
    else:
        rows, provenance = collect_cells(config, args)
    fit = fit_affine_shape(rows, min_pixels=args.min_pixels)
    training = calibrated_training_config(config, fit)
    training['train_url'] = str(output / 'training')
    report = {'fit': fit, 'provenance': provenance, 'min_pixels': args.min_pixels,
              'input_config': str(config_path), 'input_config_sha256': file_hash(config_path),
              'objective_scope': 'in-sample residual-variance fit; no held-out or PSNR claim'}
    output.mkdir(parents=True, exist_ok=True)
    fields = ['group', 'mean', 'variance', 'count']
    fields += sorted(set().union(*(set(row) for row in rows)) - set(fields))
    with (output / 'cells.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'fit.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (output / 'train.yaml').open('x', encoding='utf-8') as handle:
        handle.write('# Fitted from noisy training statistics; see fit.json in this directory.\n')
        yaml.safe_dump(training, handle, sort_keys=False)
    print('Fitted a={:.10g}, b={:.10g}, epsilon={}'.format(fit['a'], fit['b'], fit['epsilon']))
    print('Training mode: {}'.format(training['train']['mode']))
    print('Generated training config: {}'.format(output / 'train.yaml'))


if __name__ == '__main__':
    main()
