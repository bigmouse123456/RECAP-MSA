"""Problem 3: interpretable predictions for Attachment 4 plus a faithfulness check.

For every sample the RobustMSA ensemble reports polarity / intensity and
  * exact Shapley contributions of text, audio and vision to the intensity.
    A modality is removed by zeroing it, which the model saw during training
    (whole-modality drops), so every coalition stays in-distribution.  The
    values are additive: baseline + sum(shapley) == full prediction.
  * occlusion importance of every text token and every 5-frame audio/vision
    window (prediction change when that piece is zeroed), and
  * the top-k key evidence per modality, mapped to seconds of the clip.

--faithfulness_samples N runs a deletion test on Attachment 2 valid: removing
the top-k evidence must change the prediction much more than removing k
random observed windows; it also compares occlusion with attention ranking.

    python explain_robust.py --checkpoints "ckpt/robust_mosei/robust_robust_v1_seed*.pth" \
        --decision_json outputs/ensemble_decision.json --bert_path <bert dir> \
        --input_dir <Attachment 4>/未对齐版本 --output_csv outputs/attachment4_explanations.csv
"""
import argparse
import csv
import importlib
import itertools
import json
import math
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch

from core.robust_data import MODALITIES, build_sample, load_pickle, load_split
from core.robust_eval import POLARITY_NAMES, decide
from predict_robust import (
    RawTextEncoder, complete_text_fields, load_models, load_tokenizer, missing_spans,
    text_tokens,
)

SUBSETS = [c for r in range(4) for c in itertools.combinations(range(3), r)]


class CompatUnpickler(pickle.Unpickler):
    """Read pickles written by NumPy 2.x (numpy._core) under NumPy 1.x."""

    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            try:
                importlib.import_module(module)
            except ModuleNotFoundError:
                module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def read_pickle(path):
    with open(path, 'rb') as handle:
        return CompatUnpickler(handle).load()


def normalize_fields(split, name):
    """Add a missing sample axis; rebuild text_bert from raw_text if malformed."""
    split = dict(split)
    shapes = {key: getattr(np.asarray(value), 'shape', None) for key, value in split.items()}
    for key in ('audio', 'vision', 'text'):
        if key in split and np.asarray(split[key]).ndim == 2:
            split[key] = np.asarray(split[key])[None]
    if 'text_bert' in split:
        value = np.asarray(split['text_bert'])
        if value.ndim == 2 and value.shape[0] == 3:
            value = value[None]
        if value.ndim == 3 and value.shape[1] == 3:
            split['text_bert'] = value
        else:
            print(f'NOTE: {name} text_bert shape {value.shape} unsupported; rebuilt from raw_text')
            split.pop('text_bert')
    return split, shapes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--decision_json', default='outputs/ensemble_decision.json')
    parser.add_argument('--bert_path', default='')
    parser.add_argument('--text_layer', type=int, default=-1)
    parser.add_argument('--input_dir', default='')
    parser.add_argument('--video_dir', default='', help='defaults to <input_dir>/videos')
    parser.add_argument('--output_csv', default='outputs/attachment4_explanations.csv')
    parser.add_argument('--output_json', default='',
                        help='full per-window importance (default: output_csv with .json)')
    parser.add_argument('--keyframe_dir', default='outputs/keyframes',
                        help='save the top vision evidence frame per sample ("" to skip)')
    parser.add_argument('--split', default='test')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--faithfulness_samples', type=int, default=0)
    parser.add_argument('--faithfulness_json', default='outputs/faithfulness.json')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


# ----------------------------------------------------------------- inference
@torch.no_grad()
def predict(models, samples, device, batch_size):
    """Ensemble intensity and class probabilities for a list of samples."""
    intensity, probs = [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        batch = {key: torch.stack([s[key] for s in chunk]).to(device) for key in chunk[0]}
        outs = [model(batch) for model, _, _ in models]
        intensity.append(torch.stack([o['intensity'] for o in outs]).mean(0).cpu())
        probs.append(torch.stack(
            [torch.softmax(o['polarity_logits'], -1) for o in outs]).mean(0).cpu())
    return torch.cat(intensity).numpy(), torch.cat(probs).numpy()


@torch.no_grad()
def full_outputs(models, sample, device):
    batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
    outs = [model(batch) for model, _, _ in models]

    def mean(key):
        return torch.stack([o[key][0] for o in outs]).mean(0).cpu().numpy()

    return {
        'gate': mean('modality_weights'),
        'unimodal': mean('unimodal_preds'),
        'attention': {m: torch.stack([o['temporal_weights'][m][0] for o in outs])
                      .mean(0).cpu().numpy() for m in MODALITIES},
    }


def occlude(sample, modality, ranges=None):
    """Zero positions (all when ranges is None) and mark them missing."""
    variant = dict(sample)
    features = sample[modality].clone()
    observed = sample[f'{modality}_observed'].clone()
    for start, end in ranges or [(0, features.shape[0])]:
        features[start:end] = 0
        observed[start:end] = False
    variant[modality] = features
    variant[f'{modality}_observed'] = observed
    return variant


# ----------------------------------------------------------- explanations
def shapley_values(models, sample, device, batch_size):
    variants = []
    for subset in SUBSETS:
        variant = sample
        for i, modality in enumerate(MODALITIES):
            if i not in subset:
                variant = occlude(variant, modality)
        variants.append(variant)
    intensity, _ = predict(models, variants, device, batch_size)
    value = dict(zip(SUBSETS, intensity))
    phi = np.zeros(3)
    for i in range(3):
        for subset in SUBSETS:
            if i in subset:
                continue
            weight = math.factorial(len(subset)) * math.factorial(2 - len(subset)) / 6
            phi[i] += weight * (value[tuple(sorted(subset + (i,)))] - value[subset])
    return phi, float(value[()]), float(value[(0, 1, 2)])


def evidence_windows(sample, modality, stride):
    """Occlusion units: text tokens without [CLS]/[SEP]; stride-frame windows."""
    real = sample[f'{modality}_real'].numpy()
    observed = sample[f'{modality}_observed'].numpy()
    length = int(real.sum())
    if modality == 'text':
        units = [(p, p + 1, p) for p in range(1, max(length - 1, 1))]
    else:
        units = [(s, min(s + stride, length), s // stride) for s in range(0, length, stride)]
    return [(a, b, step) for a, b, step in units if observed[a:b].any()]


def occlusion_importance(models, sample, full_intensity, strides, device, batch_size):
    windows, variants = {}, []
    for modality in MODALITIES:
        windows[modality] = evidence_windows(sample, modality, strides[modality])
        variants += [occlude(sample, modality, [(a, b)]) for a, b, _ in windows[modality]]
    intensity, _ = predict(models, variants, device, batch_size) if variants else ([], None)
    importance, offset = {}, 0
    for modality in MODALITIES:
        count = len(windows[modality])
        importance[modality] = full_intensity - np.asarray(intensity[offset:offset + count])
        offset += count
    return windows, importance


def whole_word(tokens, position):
    start = position
    while start > 0 and tokens[start].startswith('##'):
        start -= 1
    end = position + 1
    while end < len(tokens) and tokens[end].startswith('##'):
        end += 1
    return tokens[start] + ''.join(t[2:] for t in tokens[start + 1:end])


def format_evidence(modality, windows, importance, top_k, length, duration, tokens):
    order = np.argsort(-np.abs(importance))[:top_k]
    items = []
    for rank in order:
        start, end, _ = windows[rank]
        effect = f'{importance[rank]:+.3f}'
        if modality == 'text':
            word = whole_word(tokens, start) if tokens and start < len(tokens) else f'token{start}'
            items.append(f'"{word}"(token {start}, {start / max(length, 1):.0%}, {effect})')
        else:
            where = f'frames {start}-{end - 1}'
            if duration:
                where += f' | {start / length * duration:.2f}-{end / length * duration:.2f}s'
            items.append(f'{where} ({effect})')
    return '; '.join(items)


# ------------------------------------------------------------------ video
def video_duration(path):
    try:
        import cv2
        capture = cv2.VideoCapture(str(path))
        fps, frames = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        if fps > 0 and frames > 0:
            return float(frames / fps)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
             str(path)], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return None


def save_keyframe(video, seconds, target):
    try:
        import cv2
        capture = cv2.VideoCapture(str(video))
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        ok, frame = capture.read()
        capture.release()
        if ok:
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), frame)
            return str(target)
    except Exception:
        pass
    return ''


# -------------------------------------------------------------- main tasks
def explain_directory(args, models, cfg, normalizer, decision, tokenizer, encoder, device):
    text_source = cfg['data']['text_source']
    strides = models[0][0].strides
    input_dir = Path(args.input_dir)
    video_dir = Path(args.video_dir) if args.video_dir else input_dir / 'videos'
    files = sorted(input_dir.glob('*.pkl'))
    if not files:
        raise FileNotFoundError(f'No PKL files in {input_dir}')

    rows, details = [], []
    for path in files:
        payload = read_pickle(path)
        split = payload[args.split] if args.split in payload else payload
        split, shapes = normalize_fields(split, path.name)
        if path == files[0]:
            print(f'Field shapes in {path.name}: {shapes}')
        split = complete_text_fields(split, text_source, encoder)
        record = load_split(split, text_source)
        for index in range(record['count']):
            sample_id = record['ids'][index] if record['count'] > 1 else path.stem
            raw = {key: value[index] for key, value in record['arrays'].items()}
            lengths = {key: int(value[index]) for key, value in record['lengths'].items()}
            sample = build_sample(raw, lengths, normalizer, text_source)
            intensity, probs = predict(models, [sample], device, args.batch_size)
            intensity, probs = float(intensity[0]), probs[0]
            polarity = int(decide(probs[None], np.array([intensity]), decision)[0])
            extra = full_outputs(models, sample, device)
            phi, baseline, full = shapley_values(models, sample, device, args.batch_size)
            windows, importance = occlusion_importance(
                models, sample, intensity, strides, device, args.batch_size)
            tokens = text_tokens(record, index, tokenizer, lengths['text'])
            video = video_dir / f'{path.stem}.mp4'
            duration = video_duration(video) if video.exists() else None
            share = np.abs(phi) / max(np.abs(phi).sum(), 1e-8)

            row = {
                'sample_id': sample_id, 'source_file': path.name,
                'video_file': video.name if video.exists() else '',
                'video_duration_s': round(duration, 3) if duration else '',
                'raw_text': record['raw_text'][index],
                'polarity_pred': POLARITY_NAMES[polarity],
                'intensity_pred': round(intensity, 4),
                'prob_negative': round(float(probs[0]), 4),
                'prob_neutral': round(float(probs[1]), 4),
                'prob_positive': round(float(probs[2]), 4),
                'main_modality': MODALITIES[int(np.argmax(np.abs(phi)))],
                'main_modality_gate': MODALITIES[int(np.argmax(extra['gate']))],
                'baseline_intensity': round(baseline, 4),
                'additivity_error': round(abs(baseline + phi.sum() - full), 6),
            }
            detail = {'sample_id': sample_id, 'duration': duration, 'tokens': tokens,
                      'lengths': lengths, 'modalities': {}}
            for i, modality in enumerate(MODALITIES):
                length = lengths[modality]
                row[f'{modality}_shapley'] = round(float(phi[i]), 4)
                row[f'{modality}_contribution_share'] = round(float(share[i]), 4)
                row[f'{modality}_gate_weight'] = round(float(extra['gate'][i]), 4)
                row[f'{modality}_unimodal_intensity'] = round(float(extra['unimodal'][i]), 4)
                row[f'{modality}_observed_ratio'] = round(float(
                    sample[f'{modality}_observed'].sum() / max(sample[f'{modality}_real'].sum(), 1)), 4)
                row[f'{modality}_evidence'] = format_evidence(
                    modality, windows[modality], importance[modality], args.top_k,
                    length, duration, tokens)
                detail['modalities'][modality] = {
                    'windows': [[a, b] for a, b, _ in windows[modality]],
                    'occlusion': [round(float(v), 5) for v in importance[modality]],
                    'attention': [round(float(extra['attention'][modality][step]), 5)
                                  for _, _, step in windows[modality]],
                    'missing_spans': missing_spans(sample[f'{modality}_observed'].numpy(),
                                                   sample[f'{modality}_real'].numpy()),
                }
            if args.keyframe_dir and duration and windows['vision']:
                best = int(np.argmax(np.abs(importance['vision'])))
                start, end, _ = windows['vision'][best]
                seconds = (start + end) / 2 / lengths['vision'] * duration
                row['vision_keyframe'] = save_keyframe(
                    video, seconds, Path(args.keyframe_dir) / f'{sample_id}_vision_top1.jpg')
                row['vision_keyframe_s'] = round(seconds, 3)
            rows.append(row)
            details.append(detail)
            print(f'{sample_id}: {row["polarity_pred"]} {intensity:+.3f} main={row["main_modality"]} '
                  f'shapley T/A/V={phi.round(3).tolist()} additivity_err={row["additivity_error"]}')

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(output, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    detail_path = Path(args.output_json) if args.output_json else output.with_suffix('.json')
    with open(detail_path, 'w', encoding='utf-8') as handle:
        json.dump(details, handle, ensure_ascii=False)
    print(f'Wrote {len(rows)} rows to {output} and per-window details to {detail_path}')


def faithfulness(args, models, cfg, normalizer, device):
    """Deletion test on Attachment 2 valid (labels unused)."""
    from scipy.stats import spearmanr

    text_source = cfg['data']['text_source']
    strides = models[0][0].strides
    record = load_pickle(cfg['data']['path'], text_source)['valid']
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(record['count'], min(args.faithfulness_samples, record['count']),
                        replace=False)
    k = args.top_k
    stats = {m: {'occlusion_top': [], 'attention_top': [], 'random': [], 'spearman': []}
             for m in MODALITIES}
    modality_stats = {'top_shapley': [], 'other': []}
    for count, index in enumerate(chosen, 1):
        raw = {key: value[index] for key, value in record['arrays'].items()}
        lengths = {key: int(value[index]) for key, value in record['lengths'].items()}
        sample = build_sample(raw, lengths, normalizer, text_source)
        base = float(predict(models, [sample], device, args.batch_size)[0][0])
        extra = full_outputs(models, sample, device)
        windows, importance = occlusion_importance(
            models, sample, base, strides, device, args.batch_size)
        phi, _, _ = shapley_values(models, sample, device, args.batch_size)

        variants, keys = [], []
        for modality in MODALITIES:
            units = windows[modality]
            if len(units) < 2 * k:
                continue
            attention = np.array([extra['attention'][modality][step] for _, _, step in units])
            picks = {
                'occlusion_top': np.argsort(-np.abs(importance[modality]))[:k],
                'attention_top': np.argsort(-attention)[:k],
                'random': rng.choice(len(units), k, replace=False),
            }
            for name, chosen_units in picks.items():
                variants.append(occlude(sample, modality, [units[u][:2] for u in chosen_units]))
                keys.append((modality, name))
            rho = spearmanr(np.abs(importance[modality]), attention).correlation
            if np.isfinite(rho):
                stats[modality]['spearman'].append(float(rho))
        top = int(np.argmax(np.abs(phi)))
        for i, modality in enumerate(MODALITIES):
            variants.append(occlude(sample, modality))
            keys.append(('modality', 'top_shapley' if i == top else 'other'))
        changed, _ = predict(models, variants, device, args.batch_size)
        for (group, name), value in zip(keys, changed):
            delta = abs(base - float(value))
            (modality_stats if group == 'modality' else stats[group])[name].append(delta)
        if count % 50 == 0:
            print(f'faithfulness: {count}/{len(chosen)} samples')

    summary = {}
    print(f'\nDeletion test on Attachment 2 valid ({len(chosen)} samples, k={k}); '
          'mean |change of intensity|')
    for modality in MODALITIES:
        s = stats[modality]
        summary[modality] = {name: round(float(np.mean(v)), 4) if v else None
                             for name, v in s.items()}
        print(f'  {modality:6s} top-k occlusion {summary[modality]["occlusion_top"]}  '
              f'top-k attention {summary[modality]["attention_top"]}  '
              f'random k {summary[modality]["random"]}  '
              f'spearman(|occlusion|, attention) {summary[modality]["spearman"]}')
    summary['modality_removal'] = {name: round(float(np.mean(v)), 4)
                                   for name, v in modality_stats.items()}
    print(f'  remove top-Shapley modality {summary["modality_removal"]["top_shapley"]} vs '
          f'other modality {summary["modality_removal"]["other"]}')
    path = Path(args.faithfulness_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(f'Wrote {path}')


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    models = load_models(args.checkpoints, device)
    cfg = models[0][2]['config']
    if cfg['data']['text_source'] != 'precomputed':
        raise NotImplementedError('explain_robust.py supports text_source=precomputed')
    if args.bert_path:
        cfg['model']['bert_pretrained'] = args.bert_path
    normalizer = models[0][1]
    with open(args.decision_json, encoding='utf-8') as handle:
        decision = json.load(handle)
    print(f'Polarity rule: {decision}')

    if args.input_dir:
        tokenizer = load_tokenizer(cfg)
        encoder = RawTextEncoder(cfg, tokenizer, args.text_layer, device) if tokenizer else None
        explain_directory(args, models, cfg, normalizer, decision, tokenizer, encoder, device)
    if args.faithfulness_samples:
        faithfulness(args, models, cfg, normalizer, device)


if __name__ == '__main__':
    main()
