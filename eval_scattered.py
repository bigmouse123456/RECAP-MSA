"""Robustness to many short missing gaps (scattered all-zero rows).

Attachment 2 audio/vision contain almost no interior all-zero rows, so the
single-frame gaps seen in the special-test files are artificial missingness.
This evaluates the ensemble on Attachment 2 valid/test after zeroing scattered
1-3 frame gaps that cover a given share of the valid audio/vision length.

    python eval_scattered.py --checkpoints "ckpt/robust_mosei/robust_robust_v1_seed*.pth"
"""
import argparse
import json

import numpy as np
import torch

from analyze_robust import ensemble_collect, load_models
from core.robust_data import FeatureNormalizer, RobustMSADataset, load_pickle
from core.robust_eval import compute_metrics


def scattered_spans(rng, length, ratio, max_gap):
    """Random 1..max_gap frame gaps until about ratio * length frames are missing."""
    target = int(round(ratio * length))
    missing = np.zeros(length, dtype=bool)
    while missing.sum() < target:
        size = int(rng.integers(1, max_gap + 1))
        start = int(rng.integers(0, max(1, length - size + 1)))
        missing[start:start + size] = True
    spans, start = [], None
    for position, value in enumerate(list(missing) + [False]):
        if value and start is None:
            start = position
        elif not value and start is not None:
            spans.append((start, position))
            start = None
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--decision_json', default='outputs/ensemble_decision.json')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--max_gap', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    models = load_models(args.checkpoints, device)
    cfg = models[0][1]['config']
    data_cfg = cfg['data']
    normalizer = FeatureNormalizer.from_state_dict(models[0][1]['normalizer'])
    records = load_pickle(data_cfg['path'], data_cfg['text_source'])
    with open(args.decision_json, encoding='utf-8') as handle:
        decision = json.load(handle)

    for split in ('valid', 'test'):
        record = records[split]
        complete = RobustMSADataset(record, normalizer, data_cfg, mode='complete')
        print(f'{split} complete           {compute_metrics(ensemble_collect(models, complete, cfg, device), decision)}')
        for ratio in args.ratios:
            rng = np.random.default_rng(2026)
            spans = [
                {modality: scattered_spans(rng, int(record['lengths'][modality][index]),
                                           ratio, args.max_gap)
                 for modality in ('audio', 'vision')}
                for index in range(record['count'])
            ]
            dataset = RobustMSADataset(record, normalizer, data_cfg, mode='missing', fixed_spans=spans)
            metrics = compute_metrics(ensemble_collect(models, dataset, cfg, device), decision)
            print(f'{split} scattered A+V {ratio:.0%}  {metrics}')


if __name__ == '__main__':
    main()
