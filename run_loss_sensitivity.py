"""Train and plot the three RobustMSA loss-weight sensitivity studies.

Run from the repository root on the training server::

    python run_loss_sensitivity.py --gpus 1 2 3 --jobs_per_gpu 1

The MAE coefficient stays fixed at 1.0. Each of the other loss coefficients is
varied one at a time; the remaining two retain their configured defaults.
Model selection and polarity calibration use Attachment 2 validation data only.
The primary figure also uses validation results, avoiding test-set tuning.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from run_ablation import run_pool  # noqa: E402


ROOT = Path(__file__).resolve().parent
PARAMS = {
    'cls': ('Classification loss weight', (0.0, 0.1, 0.25, 0.5, 1.0)),
    'unimodal': ('Unimodal auxiliary loss weight', (0.0, 0.1, 0.2, 0.4, 0.8)),
    'corr': ('Pearson loss weight', (0.0, 0.05, 0.1, 0.2, 0.4)),
}
METRICS = (
    ('Polarity_Accuracy', 'Polarity accuracy ↑'),
    ('Polarity_Macro_F1', 'Polarity macro-F1 ↑'),
    ('MAE', 'Intensity MAE ↓'),
    ('Corr', 'Pearson correlation ↑'),
)
VIEWS = (('complete', 'Original input', '#2166ac'),
         ('missing', 'Fixed synthetic missing input', '#b35806'))


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default='configs/robust_mosei.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333])
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--jobs_per_gpu', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='outputs/loss_sensitivity')
    parser.add_argument('--log_dir', default='logs/loss_sensitivity')
    parser.add_argument('--checkpoint_dir', default='ckpt/loss_sensitivity')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    parser.add_argument('--plot_test', action='store_true',
                        help='also draw a descriptive test figure; never select weights on it')
    return parser.parse_args()


def key_for(param, value, defaults):
    if value == defaults[param]:
        return 'default'
    return f'{param}_{format(value, "g").replace(".", "p")}'


def experiment_grid(defaults):
    points = []
    runs = {'default': {}}
    for param, (_, values) in PARAMS.items():
        if defaults[param] not in values:
            raise ValueError(f'Configured loss.{param}={defaults[param]} is absent from grid {values}')
        for value in values:
            key = key_for(param, value, defaults)
            runs.setdefault(key, {} if key == 'default' else {f'loss.{param}': value})
            points.append((param, value, key))
    return runs, points


def manifest_for(config_path, cfg, seeds, runs):
    return {
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'data_path': cfg['data']['path'],
        'seeds': seeds,
        'runs': runs,
        'note': 'Attachment 2 train; checkpoint and polarity rule selected on valid only',
    }


def check_manifest(path, expected, create):
    if path.exists():
        actual = json.loads(path.read_text(encoding='utf-8'))
        if actual != expected:
            raise RuntimeError(f'{path} describes another experiment; choose a new --output_dir')
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')


def train_jobs(args, runs):
    jobs = []
    output = ROOT / args.output_dir
    for key, overrides in runs.items():
        for seed in args.seeds:
            result = output / 'runs' / f'{key}_seed{seed}.json'
            if result.exists():
                continue
            ckpt = ROOT / args.checkpoint_dir / key
            command = [sys.executable, '-u', str(ROOT / 'train_robust.py'),
                       '--config_file', str(ROOT / args.config_file),
                       '--seed', str(seed), '--device', args.device,
                       '--result_json', str(result),
                       '--set', f'checkpoint_dir={json.dumps(str(ckpt))}',
                       '--set', 'checkpoint_tag="loss_sensitivity"',
                       '--set', f'data.num_workers={args.num_workers}']
            for name, value in overrides.items():
                command += ['--set', f'{name}={json.dumps(value)}']
            jobs.append((f'{key}_seed{seed}', command,
                         ROOT / args.log_dir / f'{key}_seed{seed}.log'))
    return jobs


def write_summary(args, points):
    output = ROOT / args.output_dir
    rows = []
    incomplete = []
    for param, value, key in points:
        paths = [output / 'runs' / f'{key}_seed{seed}.json' for seed in args.seeds]
        if not all(path.exists() for path in paths):
            incomplete.append((param, value))
            continue
        results = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
        for seed, result in zip(args.seeds, results):
            if result['seed'] != seed:
                raise ValueError(f'Seed mismatch in {key}: expected {seed}, got {result["seed"]}')
        for split in ('valid', 'test'):
            row = {'param': param, 'param_label': PARAMS[param][0], 'value': value,
                   'is_default': key == 'default', 'split': split,
                   'n_seeds': len(args.seeds)}
            for view, _, _ in VIEWS:
                for metric, _ in METRICS:
                    values = [float(result[split][view][metric]) for result in results]
                    row[f'{view}_{metric}_mean'] = round(float(np.mean(values)), 4)
                    row[f'{view}_{metric}_std'] = round(float(np.std(values)), 4)
            rows.append(row)
    if rows:
        path = output / 'loss_weight_summary.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {path}')
    if incomplete:
        print(f'Incomplete points ({len(incomplete)}): {incomplete}; figures wait for all seeds')
    return rows, incomplete


def plot_figure(rows, defaults, split, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(METRICS), len(PARAMS), figsize=(14, 12), squeeze=False)
    for col, (param, (label, values)) in enumerate(PARAMS.items()):
        points = sorted((row for row in rows if row['param'] == param and row['split'] == split),
                        key=lambda row: float(row['value']))
        if len(points) != len(values):
            raise ValueError(f'{param}: expected {len(values)} complete points, found {len(points)}')
        x = np.arange(len(points))
        for row_index, (metric, axis_label) in enumerate(METRICS):
            ax = axes[row_index][col]
            for view, view_label, color in VIEWS:
                mean = np.array([float(p[f'{view}_{metric}_mean']) for p in points])
                std = np.array([float(p[f'{view}_{metric}_std']) for p in points])
                ax.plot(x, mean, '-o', color=color, linewidth=1.8, markersize=4,
                        label=view_label)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
            default_index = next(i for i, p in enumerate(points)
                                 if float(p['value']) == defaults[param])
            ax.axvline(default_index, color='#777777', linestyle='--', linewidth=0.9)
            ax.set_xticks(x, [format(float(p['value']), 'g') for p in points])
            ax.grid(axis='y', alpha=0.2)
            if col == 0:
                ax.set_ylabel(axis_label)
            if row_index == 0:
                ax.set_title(label)
            if row_index == len(METRICS) - 1:
                ax.set_xlabel('Loss weight (dashed = default)')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f'Loss-weight sensitivity — Attachment 2 {split} (mean ± SD, '
                 f'{rows[0]["n_seeds"]} seeds; one factor at a time)', y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}')


def main():
    args = args_parser()
    if args.jobs_per_gpu < 1 or not args.gpus or not args.seeds:
        raise ValueError('At least one GPU, one seed, and one job per GPU are required')
    config_path = ROOT / args.config_file
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    defaults = {param: float(cfg['loss'][param]) for param in PARAMS}
    runs, points = experiment_grid(defaults)
    output = ROOT / args.output_dir
    expected = manifest_for(config_path, cfg, args.seeds, runs)
    check_manifest(output / 'experiment.json', expected, create=False)
    jobs = train_jobs(args, runs)
    print(f'{len(runs)} configurations, {len(args.seeds)} seeds, {len(jobs)} runs remaining')
    if args.dry_run:
        for name, command, _ in jobs:
            print(name, subprocess.list2cmdline(command))
        return
    if not args.summarize_only:
        check_manifest(output / 'experiment.json', expected, create=True)
        failures = run_pool(jobs, args.gpus, args.jobs_per_gpu)
        if failures:
            print(f'Failed runs (inspect logs and rerun): {failures}')
    rows, incomplete = write_summary(args, points)
    if not incomplete:
        plot_figure(rows, defaults, 'valid', output / 'loss_weight_valid.png')
        if args.plot_test:
            plot_figure(rows, defaults, 'test', output / 'loss_weight_test.png')


if __name__ == '__main__':
    main()
