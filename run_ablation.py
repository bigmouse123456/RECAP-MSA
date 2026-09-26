"""Ablation study and key-parameter sensitivity for RobustMSA.

    python run_ablation.py --gpus 1 2 --jobs_per_gpu 2        # train + summarize
    python run_ablation.py --dry_run                          # list the jobs
    python run_ablation.py --summarize_only                   # rebuild tables

Every configuration is trained with the same seeds on the Attachment 2 train
split, selected on valid and evaluated once on test, exactly like the main
model (train_robust.py).  Finished runs are skipped, so the command can be
re-run after an interruption.  Outputs:

    outputs/ablation/runs/<key>_seed<s>.json    per-run metrics
    outputs/ablation/ablation_summary.csv       ablation, mean/std over seeds
    outputs/ablation/ablation_table.md          ablation table for the paper (test)
    outputs/ablation/param_summary.csv          parameter sensitivity
    logs/ablation/<key>_seed<s>.log             training logs
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

# key -> (Chinese label, English label, overrides)
ABLATIONS = {
    'full': ('完整模型', 'Full model', {}),
    'no_aug': ('w/o 缺失增强（连续片段+整模态）', 'w/o missing augmentation',
               {'data.augment.span_prob': 0, 'data.augment.modality_drop_prob': 0}),
    'no_modality_drop': ('w/o 整模态丢弃', 'w/o whole-modality drop',
                         {'data.augment.modality_drop_prob': 0}),
    'no_norm': ('w/o 特征标准化', 'w/o feature standardization', {'data.normalize': False}),
    'no_missing_mask': ('w/o 缺失掩码（零行当作观测）', 'w/o missing mask',
                        {'model.missing_mask': False}),
    'mean_fusion': ('w/o 可靠性门控（等权融合）', 'w/o reliability gate (mean fusion)',
                    {'model.fusion': 'mean'}),
    'no_unimodal': ('w/o 单模态辅助损失', 'w/o unimodal auxiliary loss', {'loss.unimodal': 0}),
    'no_corr': ('w/o Pearson 相关损失', 'w/o Pearson loss', {'loss.corr': 0}),
    'no_cls': ('w/o 分类损失（极性由强度阈值给出）', 'w/o classification loss', {'loss.cls': 0}),
}
# Evaluation-only rows computed from the full-model runs (no extra training).
EVAL_ROWS = {
    'full_argmax': ('w/o 验证集极性校准（分类头直接 argmax）', 'w/o polarity calibration'),
    'full_ensemble': ('完整模型 + 三模型集成（提交版本）', 'Full model, 3-seed ensemble'),
}
# name -> (config keys set together, values, Chinese label, English label)
PARAMS = {
    'max_ratio': (['data.augment.max_ratio'], [0.2, 0.4, 0.6, 0.8],
                  '训练缺失比例上限', 'Max training missing ratio'),
    'cls_weight': (['loss.cls'], [0.1, 0.25, 0.5, 1.0, 2.0],
                   '分类损失权重', 'Classification loss weight'),
    'hidden_dim': (['model.hidden_dim'], [32, 64, 128, 256], '隐藏维度', 'Hidden dimension'),
    'stride': (['model.downsample.audio', 'model.downsample.vision'], [2, 5, 10, 20],
               '音视频池化步长（帧）', 'Audio/vision pooling stride (frames)'),
}
SPLITS, VIEWS = ('valid', 'test'), ('complete', 'missing')
METRICS = ('MAE', 'Corr', 'Polarity_Accuracy', 'Polarity_Macro_F1')
OUT, LOGS = Path('outputs/ablation'), Path('logs/ablation')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/robust_mosei.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333])
    parser.add_argument('--gpus', nargs='+', default=['0'],
                        help='GPU ids for CUDA_VISIBLE_DEVICES, e.g. --gpus 1 2')
    parser.add_argument('--jobs_per_gpu', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers per run')
    parser.add_argument('--only', nargs='*', default=None,
                        help='restrict to these ablation keys / parameter names')
    parser.add_argument('--extra_set', action='append', default=[],
                        help='extra KEY=VALUE for every run (e.g. optim.epochs=2 for a smoke test)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    return parser.parse_args()


def get(cfg, dotted):
    node = cfg
    for part in dotted.split('.'):
        node = node[part]
    return node


def build_runs(cfg, only):
    """Unique run keys -> overrides.  A parameter at its default value reuses 'full'."""
    runs, param_points = {}, []
    for key, (_, _, overrides) in ABLATIONS.items():
        if only is None or key in only or (key == 'full' and only):
            runs[key] = overrides
    for name, (keys, values, _, _) in PARAMS.items():
        if only is not None and name not in only:
            continue
        default = get(cfg, keys[0])
        for value in values:
            key = 'full' if value == default else f'{name}_{value}'
            runs.setdefault(key, {k: value for k in keys})
            param_points.append((name, value, key, value == default))
    return runs, param_points


def run_path(key, seed):
    return OUT / 'runs' / f'{key}_seed{seed}.json'


def train_command(args, key, seed, overrides):
    command = [sys.executable, 'train_robust.py', '--config_file', args.config_file,
               '--seed', str(seed), '--device', args.device,
               '--result_json', str(run_path(key, seed))]
    settings = {'checkpoint_dir': f'ckpt/ablation/{key}', 'checkpoint_tag': 'abl',
                'data.num_workers': args.num_workers, **overrides}
    for name, value in settings.items():
        command += ['--set', f'{name}={json.dumps(value)}']
    for extra in args.extra_set:
        command += ['--set', extra]
    return command


def ensemble_command(args, key):
    return [sys.executable, 'analyze_robust.py',
            '--checkpoints', f'ckpt/ablation/{key}/robust_abl_seed*.pth', '--skip_patterns',
            '--decision_json', str(OUT / 'ensemble' / f'{key}_decision.json'),
            '--metrics_json', str(OUT / 'ensemble' / f'{key}.json'), '--device', args.device]


def run_pool(jobs, gpus, jobs_per_gpu):
    """jobs: list of (name, command, log_path).  Returns names that failed."""
    slots = [gpu for gpu in gpus for _ in range(jobs_per_gpu)]
    pending, running, failed, done = list(jobs), {}, [], 0
    total, start = len(jobs), time.time()
    try:
        while pending or running:
            free = [slot for slot in range(len(slots)) if slot not in running]
            while pending and free:
                slot = free.pop(0)
                name, command, log_path = pending.pop(0)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = open(log_path, 'w', encoding='utf-8')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slots[slot]), PYTHONUNBUFFERED='1')
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
                running[slot] = (name, process, handle, log_path)
                print(f'  start  {name}  (GPU {slots[slot]})', flush=True)
            time.sleep(5)
            for slot, (name, process, handle, log_path) in list(running.items()):
                if process.poll() is None:
                    continue
                handle.close()
                del running[slot]
                done += 1
                elapsed = (time.time() - start) / 60
                status = 'ok' if process.returncode == 0 else f'FAILED (see {log_path})'
                if process.returncode != 0:
                    failed.append(name)
                print(f'  [{done}/{total}] {name} {status}  {elapsed:.1f} min elapsed', flush=True)
    except KeyboardInterrupt:
        for name, process, handle, _ in running.values():
            process.terminate()
            handle.close()
        raise
    return failed


# ----------------------------------------------------------------- summary
def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def aggregate(results, field):
    """mean/std over seeds of results[i][field][view][metric] (field: valid/test/test_argmax)."""
    row = {'n_seeds': len(results)}
    for view in VIEWS:
        for metric in METRICS:
            values = [r[field][view][metric] for r in results]
            row[f'{view}_{metric}_mean'] = round(float(np.mean(values)), 4)
            row[f'{view}_{metric}_std'] = round(float(np.std(values)), 4)
    return row


def single(metrics_by_view, n):
    row = {'n_seeds': n}
    for view in VIEWS:
        for metric in METRICS:
            row[f'{view}_{metric}_mean'] = metrics_by_view[view][metric]
            row[f'{view}_{metric}_std'] = 0.0
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')


def summarize(args, runs, param_points):
    results = {}
    for key in runs:
        found = [load_json(run_path(key, s)) for s in args.seeds if run_path(key, s).exists()]
        if found:
            results[key] = found
    missing = [k for k in runs if len(results.get(k, [])) < len(args.seeds)]
    if missing:
        print(f'NOTE: incomplete seeds for {missing}; their rows use the finished seeds only')

    rows = []
    for key, (zh, en, overrides) in ABLATIONS.items():
        if key not in results:
            continue
        for split in SPLITS:
            rows.append({'variant': key, 'label_zh': zh, 'label_en': en, 'split': split,
                         'evaluation': 'single-model mean over seeds',
                         'overrides': json.dumps(overrides), **aggregate(results[key], split)})
    if 'full' in results:
        for split in SPLITS:
            rows.append({'variant': 'full_argmax', 'label_zh': EVAL_ROWS['full_argmax'][0],
                         'label_en': EVAL_ROWS['full_argmax'][1], 'split': split,
                         'evaluation': 'single-model mean over seeds, argmax polarity',
                         'overrides': '{}', **aggregate(results['full'], f'{split}_argmax')})
        ensemble_path = OUT / 'ensemble' / 'full.json'
        if ensemble_path.exists():
            ensemble = load_json(ensemble_path)
            for split in SPLITS:
                rows.append({'variant': 'full_ensemble', 'label_zh': EVAL_ROWS['full_ensemble'][0],
                             'label_en': EVAL_ROWS['full_ensemble'][1], 'split': split,
                             'evaluation': 'ensemble of seeds', 'overrides': '{}',
                             **single(ensemble[split], ensemble['checkpoints'])})
    if rows:
        write_csv(OUT / 'ablation_summary.csv', rows)
        write_table(rows)

    param_rows = []
    for name, value, key, is_default in param_points:
        if key not in results:
            continue
        _, _, zh, en = PARAMS[name]
        for split in SPLITS:
            param_rows.append({'param': name, 'label_zh': zh, 'label_en': en, 'value': value,
                               'is_default': is_default, 'split': split,
                               **aggregate(results[key], split)})
    if param_rows:
        write_csv(OUT / 'param_summary.csv', param_rows)


def write_table(rows):
    test = [r for r in rows if r['split'] == 'test']
    header = ('| 模型 | 完整 MAE↓ | 完整 Corr↑ | 完整 Acc↑ | 完整 F1↑ | '
              '缺失 MAE↓ | 缺失 Corr↑ | 缺失 Acc↑ | 缺失 F1↑ |')
    seeds = max(row['n_seeds'] for row in test if row['variant'] != 'full_ensemble')
    lines = [f'附件2测试集，均值 ± 标准差（{seeds} 个随机种子）；"缺失"为固定种子的连续片段缺失版本'
             '（所有变体使用同一缺失测试集）。',
             '', header, '|' + '---|' * 9]
    for row in test:
        cells = []
        for view in VIEWS:
            for metric in METRICS:
                mean, std = row[f'{view}_{metric}_mean'], row[f'{view}_{metric}_std']
                cells.append(f'{mean:.4f}' if row['variant'] == 'full_ensemble'
                             else f'{mean:.4f}±{std:.4f}')
        lines.append(f'| {row["label_zh"]} | ' + ' | '.join(cells) + ' |')
    path = OUT / 'ablation_table.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {path}')


def main():
    args = parse_args()
    with open(args.config_file, encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    runs, param_points = build_runs(cfg, args.only)

    if not args.summarize_only:
        jobs = [(f'{key}_seed{seed}', train_command(args, key, seed, overrides),
                 LOGS / f'{key}_seed{seed}.log')
                for key, overrides in runs.items() for seed in args.seeds
                if not run_path(key, seed).exists()]
        print(f'{len(runs)} configurations x {len(args.seeds)} seeds; '
              f'{len(jobs)} training runs to do on GPU(s) {args.gpus} '
              f'({args.jobs_per_gpu} per GPU)')
        if args.dry_run:
            for name, command, _ in jobs:
                print(f'  {name}: ' + ' '.join(command[1:]))
            return
        failed = run_pool(jobs, args.gpus, args.jobs_per_gpu)

        ensemble_keys = [k for k in runs if k in ABLATIONS
                         and all(run_path(k, s).exists() for s in args.seeds)
                         and not (OUT / 'ensemble' / f'{k}.json').exists()]
        if ensemble_keys:
            print(f'Ensemble evaluation for {ensemble_keys}')
            failed += run_pool([(f'ensemble_{k}', ensemble_command(args, k),
                                 LOGS / f'ensemble_{k}.log') for k in ensemble_keys],
                               args.gpus, args.jobs_per_gpu)
        if failed:
            print(f'FAILED: {failed}  (re-run the same command to retry only these)')
    summarize(args, runs, param_points)


if __name__ == '__main__':
    main()
