"""Attachment 3 / 4 inference with RobustMSA, including explanation fields.

    python predict_robust.py --checkpoints ckpt/robust_mosei/robust_robust_v1_seed*.pth \
        --input_dir data/attachment3 --output_csv outputs/attachment3_predictions.csv

Several checkpoints (e.g. different seeds) are averaged as an ensemble.  Each
PKL may hold one or many samples in the Attachment 2 layout ({'test': {...}}).
"""
import argparse
import csv
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from core.robust_data import FeatureNormalizer, MODALITIES, build_sample, load_split
from core.robust_eval import POLARITY_NAMES, decide
from models.robust_msa import RobustMSA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--split', default='test', help='top-level key inside each PKL')
    parser.add_argument('--decision_json', default='',
                        help='ensemble polarity rule written by analyze_robust.py')
    parser.add_argument('--text_layer', type=int, default=-1,
                        help='BERT hidden layer that reproduces Attachment 2 text '
                             '(see check_text_features.py)')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def load_models(paths, device):
    models = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            checkpoint = torch.load(path, map_location=device)
            model = RobustMSA(checkpoint['config']).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            normalizer = FeatureNormalizer.from_state_dict(checkpoint['normalizer'])
            models.append((model, normalizer, checkpoint))
            print(f'Loaded {path} (epoch {checkpoint["epoch"]}, decision {checkpoint["decision"]})')
    return models


def load_tokenizer(cfg):
    try:
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained(cfg['model']['bert_pretrained'])
    except Exception as error:  # evidence then falls back to token positions
        print(f'WARNING: tokenizer unavailable ({error}); text evidence uses positions only')
        return None


class RawTextEncoder:
    """Rebuild text_bert / 50x768 text features from raw_text.

    Attachments 3/4 only provide raw_text.  check_text_features.py verifies on
    Attachment 2 that this reproduces the stored fields (tokens and layer).
    """

    def __init__(self, cfg, tokenizer, layer, device):
        from transformers import BertModel
        self.tokenizer = tokenizer
        self.max_len = cfg['model']['max_len']['text']
        self.layer = layer
        self.device = device
        self.model = BertModel.from_pretrained(
            cfg['model']['bert_pretrained'], output_hidden_states=True
        ).to(device).eval()

    def __call__(self, raw_texts):
        encoded = self.tokenizer(
            [str(text) for text in raw_texts], max_length=self.max_len,
            padding='max_length', truncation=True, return_token_type_ids=True,
        )
        text_bert = np.stack([encoded['input_ids'], encoded['attention_mask'],
                              encoded['token_type_ids']], axis=1).astype(np.int64)
        with torch.no_grad():
            tensors = [torch.from_numpy(text_bert[:, i]).to(self.device) for i in range(3)]
            hidden = self.model(input_ids=tensors[0], attention_mask=tensors[1],
                                token_type_ids=tensors[2]).hidden_states[self.layer]
        return text_bert, hidden.cpu().numpy().astype(np.float32)


def complete_text_fields(split, text_source, encoder):
    """Add text_bert / text built from raw_text when the file lacks them."""
    need_features = text_source == 'precomputed' and 'text' not in split
    need_tokens = 'text_bert' not in split
    if not (need_features or need_tokens):
        return split
    if encoder is None:
        raise ValueError('File lacks text features; transformers/BERT is required to '
                         'rebuild them from raw_text')
    split = dict(split)
    text_bert, features = encoder(np.asarray(split['raw_text']).reshape(-1))
    split.setdefault('text_bert', text_bert)
    if need_features:
        split['text'] = features
    return split


def text_tokens(record, index, tokenizer, length):
    if tokenizer is None:
        return None
    if 'text_bert' in record['arrays']:
        ids = record['arrays']['text_bert'][index][0][:length]
    else:
        ids = tokenizer.encode(record['raw_text'][index], max_length=length, truncation=True)
    return tokenizer.convert_ids_to_tokens([int(value) for value in ids])


def describe_evidence(weights, stride, length, top_k, tokens=None):
    order = np.argsort(-weights)[:top_k]
    items = []
    for step in order:
        if weights[step] <= 0:
            continue
        start, end = step * stride, min((step + 1) * stride, length)
        where = f'{start}' if stride == 1 else f'{start}-{end - 1}'
        label = f'{tokens[step]}@' if tokens is not None and step < len(tokens) else ''
        items.append(f'{label}{where}({start / max(length, 1):.0%},w={weights[step]:.3f})')
    return '; '.join(items)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    models = load_models(args.checkpoints, device)
    cfg = models[0][2]['config']
    text_source = cfg['data']['text_source']
    tokenizer = load_tokenizer(cfg)
    decision = models[0][2]['decision']
    if args.decision_json:
        with open(args.decision_json, encoding='utf-8') as handle:
            decision = json.load(handle)
    elif len(models) > 1:
        print('WARNING: ensemble uses the first checkpoint\'s polarity rule; '
              'run analyze_robust.py and pass --decision_json')
    print(f'Polarity rule: {decision}')
    encoder = RawTextEncoder(cfg, tokenizer, args.text_layer, device) if tokenizer else None

    files = sorted(Path(args.input_dir).glob('*.pkl'))
    if not files:
        raise FileNotFoundError(f'No PKL files in {args.input_dir}')

    rows = []
    for path in files:
        with open(path, 'rb') as handle:
            payload = pickle.load(handle)
        split = payload[args.split] if args.split in payload else payload
        if 'text' not in split and text_source == 'precomputed':
            print(f'NOTE: {path.name} has no text features; rebuilt from raw_text '
                  f'(BERT hidden layer {args.text_layer})')
        split = complete_text_fields(split, text_source, encoder)
        record = load_split(split, text_source)
        if 'audio_lengths' not in split or 'vision_lengths' not in split:
            print(f'NOTE: {path.name} has no explicit lengths; inferred from last non-zero row')
        for index in range(record['count']):
            raw = {key: value[index] for key, value in record['arrays'].items()}
            lengths = {key: int(value[index]) for key, value in record['lengths'].items()}
            outputs = []
            for model, normalizer, _ in models:
                sample = build_sample(raw, lengths, normalizer, text_source)
                batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
                with torch.no_grad():
                    outputs.append(model(batch))

            def mean(key):
                return torch.stack([out[key][0] for out in outputs]).mean(0).cpu().numpy()

            intensity = float(mean('intensity'))
            probs = np.mean([torch.softmax(out['polarity_logits'][0], -1).cpu().numpy()
                             for out in outputs], axis=0)
            polarity = int(decide(probs[None], np.array([intensity]), decision)[0])
            weights = mean('modality_weights')
            unimodal = mean('unimodal_preds')
            reliability = mean('reliability')
            sample_id = record['ids'][index] if record['count'] > 1 else path.stem
            row = {
                'sample_id': sample_id,
                'source_file': path.name,
                'raw_text': record['raw_text'][index],
                'polarity_pred': POLARITY_NAMES[polarity],
                'intensity_pred': round(intensity, 4),
                'prob_negative': round(float(probs[0]), 4),
                'prob_neutral': round(float(probs[1]), 4),
                'prob_positive': round(float(probs[2]), 4),
                'main_modality': MODALITIES[int(np.argmax(weights))],
            }
            for i, modality in enumerate(MODALITIES):
                stride = models[0][0].strides[modality]
                temporal = np.mean(
                    [out['temporal_weights'][modality][0].cpu().numpy() for out in outputs], axis=0
                )
                tokens = text_tokens(record, index, tokenizer, lengths['text']) if modality == 'text' else None
                row[f'{modality}_weight'] = round(float(weights[i]), 4)
                row[f'{modality}_intensity'] = round(float(unimodal[i]), 4)
                row[f'{modality}_observed_ratio'] = round(float(reliability[i]), 4)
                row[f'{modality}_length'] = lengths[modality]
                row[f'{modality}_evidence'] = describe_evidence(
                    temporal, stride, lengths[modality], args.top_k, tokens
                )
            rows.append(row)
            print(f'{sample_id}: {row["polarity_pred"]} {intensity:+.3f} main={row["main_modality"]}')

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {output}')


if __name__ == '__main__':
    main()
