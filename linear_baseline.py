"""Small-data baseline and inference for the competition's unaligned PKLs.

Only Attachment 2 train labels fit the model. Validation selects feature groups,
regularization and the polarity decision. The test split is evaluated only
after selection. This deliberately has no PyTorch dependency.
"""
import argparse
import csv
import json
import pickle
import struct
from pathlib import Path

import numpy as np


GROUPS = {
    'text_mean': slice(0, 768),
    'text_cls': slice(768, 1536),
    'audio': slice(1536, 1610),
    'vision': slice(1610, 1645),
    'reliability': slice(1645, 1648),
    'text_reliability': slice(1645, 1646),
    'audio_reliability': slice(1646, 1647),
    'vision_reliability': slice(1647, 1648),
}
FEATURE_SETS = {
    'text_mean': ('text_mean',),
    'text_cls': ('text_cls',),
    'text_mean_av': ('text_mean', 'audio', 'vision', 'reliability'),
    'text_cls_av': ('text_cls', 'audio', 'vision', 'reliability'),
}
POLARITIES = ('Negative', 'Neutral', 'Positive')


def _summary(rows, length):
    rows = np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)
    real = rows[:max(1, min(int(length), len(rows)))]
    valid = np.any(real != 0, axis=1)
    if not valid.any():
        return np.zeros(rows.shape[1], dtype=np.float32), 0.0
    return real[valid].mean(axis=0), float(valid.mean())


def features(split):
    """Training and special-test files use the same observed-row rule."""
    count = len(split['audio'])
    result = np.zeros((count, 1648), dtype=np.float32)
    for i in range(count):
        text = split['text'][i]
        if 'text_bert' in split:
            text_length = int(np.asarray(split['text_bert'][i])[1].sum())
        else:
            text_length = len(text)
        mean, ratio = _summary(text, text_length)
        result[i, GROUPS['text_mean']] = mean
        # [CLS] can itself be missing; use the first observed token as fallback.
        real_text = np.asarray(text[:max(1, text_length)])
        observed = np.flatnonzero(np.any(real_text != 0, axis=1))
        if len(observed):
            result[i, GROUPS['text_cls']] = real_text[observed[0]]
        result[i, 1645] = ratio
        for modality, ratio_index in (('audio', 1646), ('vision', 1647)):
            values = split[modality][i]
            length = (split[f'{modality}_lengths'][i]
                      if f'{modality}_lengths' in split else len(values))
            mean, ratio = _summary(values, length)
            result[i, GROUPS[modality]] = mean
            result[i, ratio_index] = ratio
    return result


def columns(names):
    return np.concatenate([np.arange(GROUPS[name].start, GROUPS[name].stop)
                           for name in names])


def normalize_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-5] = 1.0
    return mean, std


def transform(x, cols, mean, std):
    return np.clip((x[:, cols] - mean) / std, -8.0, 8.0).astype(np.float64)


def fit_ridge(x, y, alpha):
    center_x = x.mean(axis=0)
    center_y = y.mean(axis=0)
    xc = x - center_x
    gram = xc.T @ xc
    gram.flat[::len(gram) + 1] += alpha
    coef = np.linalg.solve(gram, xc.T @ (y - center_y))
    return coef, center_y - center_x @ coef


def mae(y, pred):
    return float(np.mean(np.abs(y - pred)))


def corr(y, pred):
    if np.std(pred) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(y, pred)[0, 1])


def confusion_metrics(truth, pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (truth.astype(int), pred.astype(int)), 1)
    accuracy = np.trace(cm) / max(1, cm.sum())
    f1 = 2 * cm.diagonal() / np.maximum(1, cm.sum(axis=0) + cm.sum(axis=1))
    return float(accuracy), float(f1.mean())


def decision_for_validation(y, classes, class_scores):
    """Fit one of two small polarity rules using validation labels only."""
    best = (-1.0, None)
    for low in np.arange(-0.6, 0.001, 0.1):
        for high in np.arange(0.0, 0.601, 0.1):
            predicted = np.where(y < low, 0, np.where(y > high, 2, 1))
            score = confusion_metrics(classes, predicted)[1]
            if score > best[0]:
                best = (score, {'kind': 'threshold', 'low': float(low),
                                'high': float(high)})
    for neutral_bias in np.arange(-0.3, 0.301, 0.05):
        for positive_bias in np.arange(-0.3, 0.301, 0.05):
            predicted = np.argmax(class_scores + [0, neutral_bias, positive_bias], axis=1)
            score = confusion_metrics(classes, predicted)[1]
            if score > best[0]:
                best = (score, {'kind': 'scores', 'bias': [0.0, float(neutral_bias),
                                                         float(positive_bias)]})
    return best[1]


def classify(y, scores, decision):
    if decision['kind'] == 'threshold':
        return np.where(y < decision['low'], 0,
                        np.where(y > decision['high'], 2, 1))
    return np.argmax(scores + np.asarray(decision['bias']), axis=1)


def consistent_intensity(raw, polarity):
    """Make the submitted continuous score obey the task's sign rule."""
    return np.where(polarity == 1, 0.0,
                    np.where(polarity == 0, np.minimum(raw, -1e-5),
                             np.maximum(raw, 1e-5)))


def load_labeled(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    result = {}
    for name in ('train', 'valid', 'test'):
        split = data[name]
        y = np.asarray(split['regression_labels'], dtype=np.float64).reshape(-1)
        classes = np.where(y < 0, 0, np.where(y > 0, 2, 1))
        if 'classification_labels' in split:
            supplied = np.asarray(split['classification_labels']).reshape(-1)
            if set(np.unique(supplied)).issubset({-1, 0, 1}):
                supplied = supplied + 1
            if not np.array_equal(supplied.astype(int), classes):
                raise ValueError(f'{name}: classification labels disagree with intensity signs')
        result[name] = (features(split), y, classes)
    return result


def train(args):
    data = load_labeled(args.data)
    train_x, train_y, train_c = data['train']
    valid_x, valid_y, valid_c = data['valid']
    onehot = np.eye(3)[train_c]
    best_reg = (-float('inf'), None)
    best_class = (-float('inf'), None)
    for name, groups in FEATURE_SETS.items():
        cols = columns(groups)
        mean, std = normalize_fit(train_x[:, cols])
        x = transform(train_x, cols, mean, std)
        v = transform(valid_x, cols, mean, std)
        # The initial validation sweep preferred the strongest regularization;
        # include larger values so the optimum is not forced to the grid edge.
        for alpha in (10.0, 100.0, 1000.0, 3000.0, 10000.0):
            coef, intercept = fit_ridge(x, np.column_stack([train_y, onehot]), alpha)
            predictions = v @ coef + intercept
            reg_score = corr(valid_y, predictions[:, 0]) - mae(valid_y, predictions[:, 0])
            raw_cls = np.argmax(predictions[:, 1:], axis=1)
            cls_score = confusion_metrics(valid_c, raw_cls)[1]
            candidate = (name, float(alpha), cols, mean, std, coef, intercept)
            print(f'{name} alpha={alpha:g} val MAE={mae(valid_y,predictions[:,0]):.4f} '
                  f'Corr={corr(valid_y,predictions[:,0]):.4f} '
                  f'raw Macro-F1={cls_score:.4f}', flush=True)
            if reg_score > best_reg[0]:
                best_reg = (reg_score, candidate)
            if cls_score > best_class[0]:
                best_class = (cls_score, candidate)
    def pack(candidate, output_slice):
        name, alpha, cols, mean, std, coef, intercept = candidate
        return {'name': name, 'alpha': alpha, 'cols': cols, 'mean': mean,
                'std': std, 'coef': coef[:, output_slice], 'intercept': intercept[output_slice]}
    reg = pack(best_reg[1], slice(0, 1))
    cls = pack(best_class[1], slice(1, 4))
    val_y = apply(valid_x, reg).reshape(-1)
    val_scores = apply(valid_x, cls)
    decision = decision_for_validation(val_y, valid_c, val_scores)
    val_pred = classify(val_y, val_scores, decision)
    val_consistent = consistent_intensity(val_y, val_pred)
    print('SELECTED validation', {'reg': reg['name'], 'cls': cls['name'],
          'MAE': mae(valid_y, val_consistent), 'Corr': corr(valid_y, val_consistent),
          'Accuracy': confusion_metrics(valid_c, val_pred)[0],
          'Macro-F1': confusion_metrics(valid_c, val_pred)[1], 'decision': decision})
    # Test labels are used only after every parameter and decision is fixed.
    test_x, test_y, test_c = data['test']
    test_pred_y = apply(test_x, reg).reshape(-1)
    test_pred_c = classify(test_pred_y, apply(test_x, cls), decision)
    test_consistent = consistent_intensity(test_pred_y, test_pred_c)
    metrics = {'MAE': mae(test_y, test_consistent), 'Corr': corr(test_y, test_consistent),
               'Accuracy': confusion_metrics(test_c, test_pred_c)[0],
               'Macro-F1': confusion_metrics(test_c, test_pred_c)[1]}
    print('TEST', metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{'reg_'+k: v for k,v in reg.items()},
                        **{'cls_'+k: v for k,v in cls.items()},
                        decision=json.dumps(decision),
                        validation=json.dumps({'MAE':mae(valid_y,val_consistent),
                            'Corr':corr(valid_y,val_consistent), 'Accuracy':confusion_metrics(valid_c,val_pred)[0],
                            'Macro-F1':confusion_metrics(valid_c,val_pred)[1]}),
                        test=json.dumps(metrics))
    print('Saved', output)


def apply(x, model):
    return transform(x, model['cols'], model['mean'], model['std']) @ model['coef'] + model['intercept']


def unpack(artifact, prefix):
    return {key: artifact[prefix+'_'+key] for key in
            ('name','alpha','cols','mean','std','coef','intercept')}


def group_contribution(x, model, group, coefficient):
    relevant = np.isin(model['cols'], columns((group,)))
    if not relevant.any():
        return 0.0
    transformed = transform(x[None], model['cols'], model['mean'], model['std'])[0]
    return float(transformed[relevant] @ coefficient[relevant])


def mp4_duration(path):
    """Read MP4 movie duration without an optional multimedia dependency."""
    if not path.is_file():
        return None
    with path.open('rb') as handle:
        file_size = path.stat().st_size
        while handle.tell() + 8 <= file_size:
            position = handle.tell()
            size, kind = struct.unpack('>I4s', handle.read(8))
            if size == 1:
                size = struct.unpack('>Q', handle.read(8))[0]
            elif size == 0:
                size = file_size - position
            if size < 8:
                break
            if kind == b'moov':
                end = position + size
                while handle.tell() + 8 <= end:
                    atom_start = handle.tell()
                    atom_size, atom_type = struct.unpack('>I4s', handle.read(8))
                    if atom_size < 8:
                        break
                    if atom_type == b'mvhd':
                        version = handle.read(1)[0]
                        handle.read(3)
                        handle.read(16 if version == 1 else 8)
                        timescale = struct.unpack('>I', handle.read(4))[0]
                        duration = struct.unpack('>Q' if version == 1 else '>I',
                                                 handle.read(8 if version == 1 else 4))[0]
                        return duration / timescale if timescale else None
                    handle.seek(atom_start + atom_size)
            handle.seek(position + size)
    return None


def temporal_evidence(split, index, modality, model, coefficient, top_k=3,
                      vocab=None, duration=None):
    """Exact change in this linear score when one observed window is removed."""
    group = 'text_mean' if modality == 'text' else modality
    relevant = np.isin(model['cols'], columns((group,)))
    if not relevant.any():
        return ''
    values = np.nan_to_num(np.asarray(split[modality][index], dtype=np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    if modality == 'text':
        length = (int(np.asarray(split['text_bert'][index])[1].sum())
                  if 'text_bert' in split else len(values))
        width = 1
    else:
        length = (int(split[f'{modality}_lengths'][index])
                  if f'{modality}_lengths' in split else len(values))
        width = 5
    length = max(1, min(length, len(values)))
    values = values[:length]
    observed = np.any(values != 0, axis=1)
    count = int(observed.sum())
    if count < 2:
        return ''
    mean = values[observed].mean(axis=0)
    group_mean = model['mean'][relevant]
    group_std = model['std'][relevant]
    group_coefficient = coefficient[relevant]
    reliability_group = f'{modality}_reliability'
    reliability_column = np.isin(model['cols'], columns((reliability_group,)))
    ratio_mean = model['mean'][reliability_column]
    ratio_std = model['std'][reliability_column]
    ratio_coefficient = coefficient[reliability_column]
    original_scaled = np.clip((mean - group_mean) / group_std, -8, 8)
    evidence = []
    for start in range(0, length, width):
        end = min(start + width, length)
        removed = values[start:end][observed[start:end]]
        if len(removed) == 0 or len(removed) == count:
            continue
        new_mean = (mean * count - removed.sum(axis=0)) / (count - len(removed))
        new_scaled = np.clip((new_mean - group_mean) / group_std, -8, 8)
        delta = float((original_scaled - new_scaled) @ group_coefficient)
        if len(ratio_coefficient):
            old_ratio = count / length
            new_ratio = (count - len(removed)) / length
            old_scaled = np.clip((old_ratio - ratio_mean) / ratio_std, -8, 8)
            new_ratio_scaled = np.clip((new_ratio - ratio_mean) / ratio_std, -8, 8)
            delta += float((old_scaled - new_ratio_scaled) @ ratio_coefficient)
        evidence.append((abs(delta), start, end, delta))
    evidence.sort(reverse=True)
    parts = []
    for _, start, end, delta in evidence[:top_k]:
        token = ''
        if modality == 'text' and 'text_bert' in split:
            token_id = int(split['text_bert'][index][0][start])
            token_text = (vocab[token_id] if vocab is not None
                          and 0 <= token_id < len(vocab) else str(token_id))
            token = f',token={token_text}'
        location = f'{start}-{end-1}@{start/length:.1%}'
        if duration is not None and modality != 'text':
            location += f'(~{start/length*duration:.2f}s)'
        parts.append(f'{location}:delta={delta:+.4f}{token}')
    return '; '.join(parts)


def predict(args):
    artifact = np.load(args.model, allow_pickle=False)
    reg, cls = unpack(artifact, 'reg'), unpack(artifact, 'cls')
    decision = json.loads(str(artifact['decision']))
    vocab = (Path(args.vocab).read_text(encoding='utf-8').splitlines()
             if args.vocab else None)
    files = sorted(Path(args.input_dir).glob('*.pkl'))
    if not files:
        raise FileNotFoundError(args.input_dir)
    if args.expected_count and len(files) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} PKL files, found {len(files)}')
    rows = []
    for path in files:
        with open(path,'rb') as handle:
            payload = pickle.load(handle)
        split = payload[args.split] if args.split in payload else payload
        x = features(split)
        raw_intensity = np.clip(apply(x, reg).reshape(-1), -3, 3)
        scores = apply(x, cls)
        polarity = classify(raw_intensity, scores, decision)
        intensity = consistent_intensity(raw_intensity, polarity)
        video_path = Path(args.video_dir) / f'{path.stem}.mp4' if args.video_dir else None
        duration = mp4_duration(video_path) if video_path else None
        for i in range(len(x)):
            predicted = int(polarity[i])
            ranking = np.argsort(scores[i] + (np.asarray(decision['bias'])
                               if decision['kind'] == 'scores' else 0))
            other = int(ranking[-2]) if ranking[-1] == predicted else int(ranking[-1])
            class_margin = cls['coef'][:, predicted] - cls['coef'][:, other]
            sample_id = (str(split['id'][i]) if 'id' in split
                         else path.stem if len(x) == 1 else f'{path.stem}_{i+1}')
            row = {'sample_id': sample_id, 'source_file': path.name,
                   'raw_text': str(split['raw_text'][i]) if 'raw_text' in split else '',
                   'polarity_pred': POLARITIES[predicted],
                   'intensity_pred': round(float(intensity[i]), 5),
                   'raw_linear_intensity': round(float(raw_intensity[i]), 5),
                   'video_duration_seconds': round(duration, 3) if duration else '',
                   'explanation_method': 'raw_linear_score_and_leave-window-out'}
            # Exact linear decomposition relative to the fitted training mean.
            for modality, names in {'text':('text_mean','text_cls'),
                                    'audio':('audio',), 'vision':('vision',)}.items():
                names = (*names, f'{modality}_reliability')
                row[f'{modality}_intensity_contribution'] = round(sum(
                    group_contribution(x[i], reg, name, reg['coef'][:,0]) for name in names
                ), 5)
                row[f'{modality}_polarity_margin_contribution'] = round(sum(
                    group_contribution(x[i], cls, name, class_margin) for name in names
                ), 5) if decision['kind'] == 'scores' else ''
                evidence_model, evidence_coef = (
                    (reg, reg['coef'][:,0]) if decision['kind'] == 'threshold'
                    else (cls, class_margin)
                )
                row[f'{modality}_key_windows'] = temporal_evidence(
                    split, i, modality, evidence_model, evidence_coef, args.top_k,
                    vocab, duration
                )
            contribution_key = ('polarity_margin_contribution'
                                if decision['kind'] == 'scores'
                                else 'intensity_contribution')
            row['main_modality'] = max(
                ('text', 'audio', 'vision'),
                key=lambda name: abs(float(row[f'{name}_{contribution_key}'] or 0)),
            )
            rows.append(row)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output,'w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(rows)} predictions to {output}')
    if args.submission_csv:
        submission = Path(args.submission_csv)
        submission.parent.mkdir(parents=True, exist_ok=True)
        fields = ('sample_id', 'source_file', 'polarity_pred', 'intensity_pred')
        with open(submission, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
        print(f'Wrote compact submission to {submission}')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--data', required=True)
    fit.add_argument('--output', required=True)
    infer = sub.add_parser('predict')
    infer.add_argument('--model', required=True)
    infer.add_argument('--input_dir', required=True)
    infer.add_argument('--output_csv', required=True)
    infer.add_argument('--submission_csv', default='')
    infer.add_argument('--split', default='test')
    infer.add_argument('--top_k', type=int, default=3)
    infer.add_argument('--vocab', default='', help='BERT vocab.txt for readable token evidence')
    infer.add_argument('--video_dir', default='', help='optional MP4s for approximate evidence times')
    infer.add_argument('--expected_count', type=int, default=0)
    args = parser.parse_args()
    (train if args.command == 'train' else predict)(args)


if __name__ == '__main__':
    main()
