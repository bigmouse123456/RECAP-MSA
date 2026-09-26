"""Train RobustMSA on Attachment 2 (train split), select on the valid split.

    python train_robust.py --config_file configs/robust_mosei.yaml --seed 1111

The test split is evaluated once, after model selection and polarity-rule
calibration on the validation split.  --set overrides config entries, e.g.
--set loss.cls=0.25 --set data.augment.span_prob=0 (used by run_ablation.py);
--result_json writes the final valid/test metrics.
"""
import argparse
import copy
import json
import math
import os
import random

import numpy as np
import torch
import yaml

from core.robust_data import (
    FeatureNormalizer, RobustMSADataset, load_pickle, make_loader,
)
from core.robust_eval import (
    DEFAULT_DECISION, calibrate_decision, collect, compute_metrics,
    merge_outputs, selection_score, to_device,
)
from models.robust_msa import RobustMSA, RobustMSALoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/robust_mosei.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a config entry (dotted key, YAML value); repeatable')
    parser.add_argument('--result_json', default='')
    parser.add_argument('--skip_test', action='store_true',
                        help='do not evaluate test (recommended for hyper-parameter search)')
    return parser.parse_args()


def apply_overrides(cfg, assignments):
    for assignment in assignments:
        key, _, value = assignment.partition('=')
        if not _:
            raise ValueError(f'--set expects KEY=VALUE, got {assignment!r}')
        node, parts = cfg, key.split('.')
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f'unknown config key {key!r}')
            node = node[part]
        node[parts[-1]] = yaml.safe_load(value)
        print(f'override {key} = {node[parts[-1]]!r}')
    return cfg


def build_loaders(cfg, records, normalizer):
    data_cfg = cfg['data']
    batch, workers = data_cfg['batch_size'], data_cfg['num_workers']
    seed = data_cfg['valid_missing_seed']
    loaders = {'train': make_loader(
        RobustMSADataset(records['train'], normalizer, data_cfg, mode='train'),
        batch, workers, shuffle=True,
    )}
    for split, offset in (('valid', 0), ('test', 1)):
        for view in ('complete', 'missing'):
            dataset = RobustMSADataset(
                records[split], normalizer, data_cfg, mode=view, seed=seed + offset
            )
            loaders[f'{split}_{view}'] = make_loader(dataset, batch * 2, workers, shuffle=False)
    return loaders


def class_weights(classes, device):
    counts = np.bincount(classes, minlength=3).astype(np.float32)
    if (counts == 0).any():
        raise ValueError(f'All three polarity classes are required; counts={counts.tolist()}')
    print(f'Polarity class counts (train): {counts.astype(int).tolist()}')
    return torch.tensor(counts.sum() / (3.0 * counts), device=device)


def build_optimizer(model, cfg):
    optim_cfg = cfg['optim']
    bert, decay, no_decay = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('text_frontend.'):
            bert.append(parameter)
        elif parameter.ndim < 2 or 'position' in name or 'token' in name or 'embedding' in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {'params': decay, 'lr': optim_cfg['lr'], 'weight_decay': optim_cfg['weight_decay']},
        {'params': no_decay, 'lr': optim_cfg['lr'], 'weight_decay': 0.0},
    ]
    if bert:
        groups.append({'params': bert, 'lr': optim_cfg['bert_lr'], 'weight_decay': 0.01})
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = cfg['optim']['epochs'] * steps_per_epoch
    warmup = max(1, cfg['optim']['warmup_epochs'] * steps_per_epoch)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, grad_clip):
    model.train()
    totals, batches = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        loss, logs = loss_fn(out, batch['regression'], batch['classes'])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: round(value / batches, 4) for key, value in totals.items()}


def evaluate_views(model, loaders, split, device, decision=DEFAULT_DECISION):
    outputs, metrics = {}, {}
    for view in ('complete', 'missing'):
        outputs[view] = collect(model, loaders[f'{split}_{view}'], device)
        metrics[view] = compute_metrics(outputs[view], decision)
    return outputs, metrics


def weighted_score(metrics, cfg):
    views = cfg['selection']['views']
    metric_weights = cfg['selection'].get('metrics')
    return sum(
        weight * selection_score(metrics[view], metric_weights)
        for view, weight in views.items()
    )


def main():
    args = parse_args()
    with open(args.config_file, encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    # Freeze the evaluation "missing" view before any --set changes the
    # training augmentation (identical to augment for the main model).
    cfg['data'].setdefault('eval_missing', copy.deepcopy(cfg['data']['augment']))
    cfg = apply_overrides(cfg, args.set)
    seed = cfg['seed'] if args.seed is None else args.seed
    setup_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'device={device} seed={seed}')

    text_source = cfg['data']['text_source']
    records = load_pickle(cfg['data']['path'], text_source)
    for split, record in records.items():
        print(f'{split}: {record["count"]} samples')
    modalities = ['audio', 'vision'] + (['text'] if text_source == 'precomputed' else [])
    if not cfg['data'].get('normalize', True):
        modalities = []  # ablation: raw feature scales
    normalizer = FeatureNormalizer().fit(records['train'], modalities)
    loaders = build_loaders(cfg, records, normalizer)

    model = RobustMSA(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable:,}')
    loss_fn = RobustMSALoss(cfg, class_weights(records['train']['classes'], device))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(loaders['train']))

    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    ckpt_path = os.path.join(
        cfg['checkpoint_dir'], f'robust_{cfg["checkpoint_tag"]}_seed{seed}.pth'
    )
    best_score, best_epoch, stale = -float('inf'), None, 0
    patience = cfg['optim']['early_stopping_patience']

    for epoch in range(1, cfg['optim']['epochs'] + 1):
        train_logs = train_epoch(
            model, loaders['train'], optimizer, scheduler, loss_fn, device,
            cfg['optim']['grad_clip'],
        )
        _, valid_metrics = evaluate_views(model, loaders, 'valid', device)
        score = weighted_score(valid_metrics, cfg)
        print(f'Epoch {epoch:03d} train {train_logs}')
        print(f'          valid complete {valid_metrics["complete"]}')
        print(f'          valid missing  {valid_metrics["missing"]}')
        print(f'          selection score {score:.4f}')
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                'state_dict': model.state_dict(),
                'config': cfg,
                'normalizer': normalizer.state_dict(),
                'epoch': epoch,
                'seed': seed,
                'valid_metrics': valid_metrics,
                'decision': DEFAULT_DECISION,
            }, ckpt_path)
            print(f'          -> new best, saved {ckpt_path}')
        else:
            stale += 1
            if stale >= patience:
                print(f'Early stopping: no improvement for {patience} epochs.')
                break

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # Polarity decision rule: chosen on validation (both views) only.
    valid_outputs, _ = evaluate_views(model, loaders, 'valid', device)
    merged = merge_outputs(list(valid_outputs.values()))
    decision_cfg = cfg['selection'].get('decision', {})
    decision, valid_polarity_score = calibrate_decision(
        merged['probs'], merged['intensity'], merged['classes'],
        f1_weight=decision_cfg.get('f1_weight', 1.0),
        accuracy_weight=decision_cfg.get('accuracy_weight', 0.0),
    )
    checkpoint['decision'] = decision
    checkpoint['valid_metrics'] = {
        view: compute_metrics(output, decision) for view, output in valid_outputs.items()
    }
    torch.save(checkpoint, ckpt_path)
    print(f'Best epoch {best_epoch}; calibrated polarity rule {decision} '
          f'(valid polarity objective {valid_polarity_score:.4f})')
    print(f'Valid (calibrated): {json.dumps(checkpoint["valid_metrics"])}')

    test_outputs, test_metrics = {}, {}
    if not args.skip_test:
        test_outputs, test_metrics = evaluate_views(model, loaders, 'test', device, decision)
        print(f'Test complete: {test_metrics["complete"]}')
        print(f'Test missing:  {test_metrics["missing"]}')

    if args.result_json:
        result = {
            'seed': seed, 'best_epoch': best_epoch, 'overrides': args.set,
            'checkpoint': ckpt_path, 'decision': decision,
            'valid': checkpoint['valid_metrics'], 'test': test_metrics,
            'valid_selection_score': weighted_score(checkpoint['valid_metrics'], cfg),
            'valid_polarity_objective': valid_polarity_score,
            # Same models with plain classifier argmax (no validation calibration).
            'valid_argmax': {v: compute_metrics(o, DEFAULT_DECISION) for v, o in valid_outputs.items()},
            'test_argmax': ({v: compute_metrics(o, DEFAULT_DECISION)
                             for v, o in test_outputs.items()} if test_outputs else {}),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
        with open(args.result_json, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
        print(f'Wrote {args.result_json}')


if __name__ == '__main__':
    main()
