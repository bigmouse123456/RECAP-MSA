"""Validation-only loss-weight tuning for RobustMSA.

This script produces both outputs requested for the paper:

1. Joint candidates (baseline / A / B / C), ranked mainly by Macro-F1 and
   accuracy while enforcing MAE/Pearson safeguards relative to the baseline.
2. One-factor-at-a-time sensitivity for classification, unimodal auxiliary,
   and Pearson loss weights, including mean +/- SD CSV data and a 4x3 figure.

Examples
--------
Run all configurations with three seeds on two GPUs::

    python run_loss_tuning.py --gpus 0 1 --jobs_per_gpu 1

List commands without training::

    python run_loss_tuning.py --dry_run

Rebuild summaries and the figure from completed JSON files::

    python run_loss_tuning.py --summarize_only

Only validation is evaluated during the search.  The test split should be
evaluated once, after the final configuration has been selected.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


COMBINATIONS = {
    'baseline': ('Baseline (0.50, 0.20, 0.10)', {}),
    'A': ('A: balanced recommendation (0.25, 0.20, 0.20)',
          {'loss.cls': 0.25, 'loss.unimodal': 0.20, 'loss.corr': 0.20}),
    'B': ('B: classification-first (0.25, 0.40, 0.20)',
          {'loss.cls': 0.25, 'loss.unimodal': 0.40, 'loss.corr': 0.20}),
    'C': ('C: interaction check (0.25, 0.40, 0.10)',
          {'loss.cls': 0.25, 'loss.unimodal': 0.40, 'loss.corr': 0.10}),
}

# These points reproduce the earlier loss-weight sensitivity analysis.
SENSITIVITY = {
    'cls': ('loss.cls', [0.0, 0.1, 0.25, 0.5, 1.0],
            'Classification loss weight'),
    'unimodal': ('loss.unimodal', [0.0, 0.1, 0.2, 0.4, 0.8],
                 'Unimodal auxiliary loss weight'),
    'corr': ('loss.corr', [0.0, 0.05, 0.1, 0.2, 0.4],
             'Pearson loss weight'),
}

VIEWS = ('complete', 'missing')
METRICS = ('Polarity_Accuracy', 'Polarity_Macro_F1', 'MAE', 'Corr')
OUT = Path('outputs/loss_tuning')
RUNS = OUT / 'runs'
LOGS = Path('logs/loss_tuning')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/robust_mosei.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333])
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--jobs_per_gpu', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--parts', nargs='+', choices=['combinations', 'sensitivity'],
                        default=['combinations', 'sensitivity'])
    parser.add_argument('--extra_set', action='append', default=[],
                        help='extra KEY=VALUE for every run; useful for smoke tests')
    parser.add_argument('--max_mae_increase', type=float, default=0.01)
    parser.add_argument('--max_corr_drop', type=float, default=0.01)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    return parser.parse_args()


def get_dotted(cfg, key):
    node = cfg
    for part in key.split('.'):
        node = node[part]
    return node


def value_key(value):
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def build_runs(cfg, parts):
    runs = {'baseline': {}}
    if 'combinations' in parts:
        for name, (_, overrides) in COMBINATIONS.items():
            runs[f'combo_{name}'] = overrides
    if 'sensitivity' in parts:
        for name, (config_key, values, _) in SENSITIVITY.items():
            default = float(get_dotted(cfg, config_key))
            for value in values:
                key = 'baseline' if np.isclose(value, default) else f'{name}_{value_key(value)}'
                runs.setdefault(key, {config_key: value})
    runs.pop('combo_baseline', None)
    return runs


def result_path(key, seed):
    return RUNS / f'{key}_seed{seed}.json'


def train_command(args, key, seed, overrides):
    command = [
        sys.executable, 'train_robust.py', '--config_file', args.config_file,
        '--seed', str(seed), '--device', args.device, '--skip_test',
        '--result_json', str(result_path(key, seed)),
    ]
    settings = {
        'checkpoint_dir': f'ckpt/loss_tuning/{key}',
        'checkpoint_tag': 'loss',
        'data.num_workers': args.num_workers,
        **overrides,
    }
    for name, value in settings.items():
        command += ['--set', f'{name}={json.dumps(value)}']
    for setting in args.extra_set:
        command += ['--set', setting]
    return command


def run_pool(jobs, gpus, jobs_per_gpu):
    slots = [gpu for gpu in gpus for _ in range(jobs_per_gpu)]
    pending, running, failed = list(jobs), {}, []
    completed, total, started = 0, len(jobs), time.time()
    try:
        while pending or running:
            free = [slot for slot in range(len(slots)) if slot not in running]
            while pending and free:
                slot = free.pop(0)
                name, command, log_path = pending.pop(0)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = open(log_path, 'w', encoding='utf-8')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slots[slot]), PYTHONUNBUFFERED='1')
                process = subprocess.Popen(
                    command, stdout=handle, stderr=subprocess.STDOUT, env=env
                )
                running[slot] = (name, process, handle, log_path)
                print(f'start {name} on GPU {slots[slot]}', flush=True)
            time.sleep(3)
            for slot, (name, process, handle, log_path) in list(running.items()):
                if process.poll() is None:
                    continue
                handle.close()
                del running[slot]
                completed += 1
                if process.returncode:
                    failed.append(name)
                    status = f'FAILED; see {log_path}'
                else:
                    status = 'ok'
                minutes = (time.time() - started) / 60
                print(f'[{completed}/{total}] {name}: {status} ({minutes:.1f} min)', flush=True)
    except KeyboardInterrupt:
        for _, process, handle, _ in running.values():
            process.terminate()
            handle.close()
        raise
    return failed


def load_results(key, seeds):
    results = []
    for seed in seeds:
        path = result_path(key, seed)
        if path.exists():
            with open(path, encoding='utf-8') as handle:
                results.append(json.load(handle))
    return results


def aggregate(key, results):
    row = {'run_key': key, 'n_seeds': len(results)}
    for view in VIEWS:
        for metric in METRICS:
            values = np.asarray([r['valid'][view][metric] for r in results], dtype=float)
            row[f'{view}_{metric}_mean'] = round(float(values.mean()), 5)
            row[f'{view}_{metric}_std'] = round(float(values.std()), 5)
    row['classification_objective'] = round(float(np.mean([
        0.6 * row[f'{view}_Polarity_Macro_F1_mean']
        + 0.4 * row[f'{view}_Polarity_Accuracy_mean'] for view in VIEWS
    ])), 6)
    row['mean_MAE'] = round(float(np.mean([
        row[f'{view}_MAE_mean'] for view in VIEWS
    ])), 6)
    row['mean_Corr'] = round(float(np.mean([
        row[f'{view}_Corr_mean'] for view in VIEWS
    ])), 6)
    return row


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')


def summarize_combinations(args, cfg, aggregated):
    baseline = aggregated.get('baseline')
    if not baseline:
        print('Combination summary skipped: baseline results are missing.')
        return
    rows = []
    for name, (label, overrides) in COMBINATIONS.items():
        key = 'baseline' if name == 'baseline' else f'combo_{name}'
        if key not in aggregated:
            continue
        row = dict(aggregated[key])
        row.update({
            'variant': name,
            'label': label,
            'loss_cls': overrides.get('loss.cls', get_dotted(cfg, 'loss.cls')),
            'loss_unimodal': overrides.get('loss.unimodal', get_dotted(cfg, 'loss.unimodal')),
            'loss_corr': overrides.get('loss.corr', get_dotted(cfg, 'loss.corr')),
        })
        row['delta_MAE_vs_baseline'] = round(row['mean_MAE'] - baseline['mean_MAE'], 6)
        row['delta_Corr_vs_baseline'] = round(row['mean_Corr'] - baseline['mean_Corr'], 6)
        row['eligible'] = (
            row['delta_MAE_vs_baseline'] <= args.max_mae_increase + 1e-12
            and row['delta_Corr_vs_baseline'] >= -args.max_corr_drop - 1e-12
        )
        rows.append(row)
    eligible = sorted(
        (row for row in rows if row['eligible']),
        key=lambda row: row['classification_objective'], reverse=True,
    )
    ranks = {row['variant']: rank for rank, row in enumerate(eligible, 1)}
    for row in rows:
        row['eligible_rank'] = ranks.get(row['variant'], '')
    rows.sort(key=lambda row: (not row['eligible'], -row['classification_objective']))
    write_csv(OUT / 'loss_combination_summary.csv', rows)
    if eligible:
        best = eligible[0]
        recommendation = {
            'recommended_variant': best['variant'],
            'label': best['label'],
            'classification_objective': best['classification_objective'],
            'mean_MAE': best['mean_MAE'],
            'mean_Corr': best['mean_Corr'],
            'constraints': {
                'max_MAE_increase_vs_baseline': args.max_mae_increase,
                'max_Corr_drop_vs_baseline': args.max_corr_drop,
            },
            'selection_data': 'Attachment 2 validation only',
        }
        with open(OUT / 'loss_recommendation.json', 'w', encoding='utf-8') as handle:
            json.dump(recommendation, handle, ensure_ascii=False, indent=2)
        print(f'recommended variant: {best["variant"]} '
              f'(objective={best["classification_objective"]:.4f})')


def sensitivity_rows(cfg, aggregated):
    rows = []
    for name, (config_key, values, label) in SENSITIVITY.items():
        default = float(get_dotted(cfg, config_key))
        for value in values:
            key = 'baseline' if np.isclose(value, default) else f'{name}_{value_key(value)}'
            if key not in aggregated:
                continue
            rows.append({
                'parameter': name, 'config_key': config_key, 'label': label,
                'value': value, 'is_default': bool(np.isclose(value, default)),
                **aggregated[key],
            })
    return rows


def plot_sensitivity(rows, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; CSV was written but the figure was skipped.')
        return
    view_style = {
        'complete': ('Original input', '#2a78d6'),
        'missing': ('Fixed synthetic missing input', '#e58b38'),
    }
    metric_rows = [
        ('Polarity_Accuracy', 'Polarity accuracy ↑'),
        ('Polarity_Macro_F1', 'Polarity macro-F1 ↑'),
        ('MAE', 'Intensity MAE ↓'),
        ('Corr', 'Pearson correlation ↑'),
    ]
    params = list(SENSITIVITY)
    fig, axes = plt.subplots(4, 3, figsize=(12, 9), squeeze=False)
    for col, name in enumerate(params):
        points = sorted(
            (row for row in rows if row['parameter'] == name),
            key=lambda row: float(row['value']),
        )
        if not points:
            continue
        x = np.asarray([float(row['value']) for row in points])
        for row_index, (metric, ylabel) in enumerate(metric_rows):
            ax = axes[row_index][col]
            for view, (label, color) in view_style.items():
                mean = np.asarray([row[f'{view}_{metric}_mean'] for row in points], dtype=float)
                std = np.asarray([row[f'{view}_{metric}_std'] for row in points], dtype=float)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
                ax.plot(x, mean, color=color, marker='o', markersize=4, label=label)
            for point in points:
                if point['is_default']:
                    ax.axvline(float(point['value']), color='#777777', linestyle=':', linewidth=1)
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(points[0]['label'], fontsize=10)
            if row_index == 3:
                ax.set_xlabel(SENSITIVITY[name][0])
    n_seeds = max((int(row['n_seeds']) for row in rows), default=0)
    fig.suptitle(
        f'Loss-weight sensitivity — Attachment 2 valid '
        f'(mean ± SD, up to {n_seeds} seeds; one factor at a time)', fontsize=12
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def summarize(args, cfg, runs):
    aggregated = {}
    for key in runs:
        results = load_results(key, args.seeds)
        if results:
            aggregated[key] = aggregate(key, results)
        if len(results) < len(args.seeds):
            print(f'NOTE: {key} has {len(results)}/{len(args.seeds)} completed seeds')
    if 'combinations' in args.parts:
        summarize_combinations(args, cfg, aggregated)
    if 'sensitivity' in args.parts:
        rows = sensitivity_rows(cfg, aggregated)
        write_csv(OUT / 'loss_sensitivity_summary.csv', rows)
        if rows:
            plot_sensitivity(rows, OUT / 'loss_weight_sensitivity_valid.png')


def main():
    args = parse_args()
    with open(args.config_file, encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    runs = build_runs(cfg, args.parts)
    if not args.summarize_only:
        jobs = [
            (f'{key}_seed{seed}', train_command(args, key, seed, overrides),
             LOGS / f'{key}_seed{seed}.log')
            for key, overrides in runs.items() for seed in args.seeds
            if not result_path(key, seed).exists()
        ]
        print(f'{len(runs)} configurations x {len(args.seeds)} seeds; '
              f'{len(jobs)} unfinished runs')
        if args.dry_run:
            for name, command, _ in jobs:
                print(f'{name}: ' + ' '.join(command[1:]))
            return
        failed = run_pool(jobs, args.gpus, args.jobs_per_gpu)
        if failed:
            print(f'FAILED: {failed}; rerun the same command to retry unfinished jobs')
    summarize(args, cfg, runs)


if __name__ == '__main__':
    main()
