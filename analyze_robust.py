"""Ensemble evaluation + missing-pattern analysis for the Problem 2 report.

    python analyze_robust.py --checkpoints ckpt/robust_mosei/robust_robust_v1_seed*.pth

1. Averages the given checkpoints, calibrates the polarity rule for the
   ensemble on the validation split and writes it to --decision_json
   (pass it to predict_robust.py).
2. Reports validation/test metrics on complete and missing views.
3. Evaluates controlled missing patterns on --split (default test):
   missing modality set x position (begin/middle/end/random) x duration ratio,
   written to --output_csv for the influence-law analysis and plots.
"""
import argparse
import csv
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from core.robust_data import (
    MODALITIES, FeatureNormalizer, RobustMSADataset, load_pickle, make_loader,
    positional_spans, sample_spans,
)
from core.robust_eval import (
    average_outputs, calibrate_decision, collect, compute_metrics, merge_outputs,
)
from models.robust_msa import RobustMSA

POSITIONS = ('begin', 'middle', 'end', 'random')
RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--split', default='test', choices=['valid', 'test'])
    parser.add_argument('--output_csv', default='outputs/missing_analysis.csv')
    parser.add_argument('--decision_json', default='outputs/ensemble_decision.json')
    parser.add_argument('--skip_patterns', action='store_true')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def load_models(patterns, device):
    models = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            checkpoint = torch.load(path, map_location=device)
            model = RobustMSA(checkpoint['config']).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            models.append((model, checkpoint))
    print(f'Ensemble of {len(models)} checkpoint(s)')
    return models


def ensemble_collect(models, dataset, cfg, device):
    loader = make_loader(dataset, cfg['data']['batch_size'] * 2, cfg['data']['num_workers'], False)
    return average_outputs([collect(model, loader, device) for model, _ in models])


def pattern_spans(record, modalities, position, ratio, seed):
    rng = np.random.default_rng(seed)
    spans = []
    for index in range(record['count']):
        sample = {}
        for modality in modalities:
            length = int(record['lengths'][modality][index])
            if position == 'random':
                sample[modality] = sample_spans(rng, length, ratio, ratio, 1)
            else:
                sample[modality] = positional_spans(length, ratio, position)
        spans.append(sample)
    return spans


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    models = load_models(args.checkpoints, device)
    cfg = models[0][1]['config']
    data_cfg = cfg['data']
    normalizer = FeatureNormalizer.from_state_dict(models[0][1]['normalizer'])
    records = load_pickle(data_cfg['path'], data_cfg['text_source'])
    seed = data_cfg['valid_missing_seed']

    views = {}
    for split, offset in (('valid', 0), ('test', 1)):
        for view in ('complete', 'missing'):
            dataset = RobustMSADataset(records[split], normalizer, data_cfg, mode=view, seed=seed + offset)
            views[(split, view)] = ensemble_collect(models, dataset, cfg, device)

    merged = merge_outputs([views[('valid', 'complete')], views[('valid', 'missing')]])
    decision, valid_f1 = calibrate_decision(merged['probs'], merged['intensity'], merged['classes'])
    Path(args.decision_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.decision_json, 'w', encoding='utf-8') as handle:
        json.dump(decision, handle)
    print(f'Ensemble polarity rule {decision} (valid Macro-F1 {valid_f1:.4f}) -> {args.decision_json}')
    for (split, view), outputs in views.items():
        print(f'{split:5s} {view:8s} {compute_metrics(outputs, decision)}')

    if args.skip_patterns:
        return
    record = records[args.split]
    rows = []
    subsets = [c for r in (1, 2, 3) for c in itertools.combinations(MODALITIES, r)]
    for subset, position, ratio in itertools.product(subsets, POSITIONS, RATIOS):
        if ratio == 1.0 and position != 'begin':
            continue  # whole-modality missing is position independent
        spans = pattern_spans(record, subset, position, ratio, seed)
        dataset = RobustMSADataset(record, normalizer, data_cfg, mode='missing', fixed_spans=spans)
        metrics = compute_metrics(ensemble_collect(models, dataset, cfg, device), decision)
        row = {'missing_modalities': '+'.join(subset), 'position': position if ratio < 1 else 'whole',
               'ratio': ratio, **metrics}
        rows.append(row)
        print(row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} missing patterns to {output}')


if __name__ == '__main__':
    main()
