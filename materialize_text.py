"""Recreate Attachment 2-style BERT text features for unlabeled PKLs.

The special-test files may contain only raw_text/audio/vision. This script
adds text (N,50,768) and text_bert (N,3,50) without changing the originals.
Use --reference_pkl to check numerical compatibility with Attachment 2 first.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast


def normalize_split(split):
    normalized = dict(split)
    text = np.asarray(split['raw_text']).reshape(-1)
    normalized['raw_text'] = text
    count = len(text)
    for key in ('audio', 'vision', 'text', 'text_bert'):
        if key not in normalized:
            continue
        values = np.asarray(normalized[key])
        if count == 1 and values.ndim == (2 if key != 'text_bert' else 2):
            values = values[None]
        if values.shape[0] != count:
            raise ValueError(f'{key}: first dimension {values.shape[0]} != {count}')
        normalized[key] = values
    for key in ('audio_lengths', 'vision_lengths', 'id'):
        if key in normalized:
            normalized[key] = np.asarray(normalized[key]).reshape(-1)
    return normalized


def encode(texts, tokenizer, model, device, batch_size=16):
    embeddings, triples = [], []
    for start in range(0, len(texts), batch_size):
        batch = [str(value) for value in texts[start:start + batch_size]]
        encoded = tokenizer(batch, padding='max_length', truncation=True,
                            max_length=50, return_tensors='pt')
        ids = encoded['input_ids']
        mask = encoded['attention_mask']
        segments = encoded.get('token_type_ids', torch.zeros_like(ids))
        with torch.inference_mode():
            output = model(input_ids=ids.to(device),
                           attention_mask=mask.to(device),
                           token_type_ids=segments.to(device)).last_hidden_state
        embeddings.append(output.cpu().numpy().astype(np.float32))
        triples.append(torch.stack([ids, mask, segments], dim=1).numpy())
    return np.concatenate(embeddings), np.concatenate(triples)


def verify_reference(reference_path, model, device):
    with open(reference_path, 'rb') as handle:
        reference = pickle.load(handle)['train']
    triples = np.asarray(reference['text_bert'][:4], dtype=np.int64)
    stored = np.asarray(reference['text'][:4], dtype=np.float32)
    with torch.inference_mode():
        result = model(
            input_ids=torch.from_numpy(triples[:, 0]).to(device),
            attention_mask=torch.from_numpy(triples[:, 1]).to(device),
            token_type_ids=torch.from_numpy(triples[:, 2]).to(device),
        ).last_hidden_state.cpu().numpy()
    active = triples[:, 1].astype(bool)
    error = np.abs(result - stored)[active]
    print(f'Attachment 2 BERT match: active-token mean_abs={error.mean():.6f}, '
          f'max_abs={error.max():.6f}')
    if error.mean() > 0.05:
        raise ValueError('This BERT checkpoint does not reproduce the training text features.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='local bert-base-uncased directory')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--reference_pkl', default='')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = BertTokenizerFast.from_pretrained(args.model, local_files_only=True)
    model = BertModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    if args.reference_pkl:
        verify_reference(args.reference_pkl, model, device)
    source = Path(args.input_dir)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob('*.pkl'))
    if not files:
        raise FileNotFoundError(source)
    for path in files:
        with open(path, 'rb') as handle:
            payload = pickle.load(handle)
        wrapped = 'test' in payload
        split = normalize_split(payload['test'] if wrapped else payload)
        if 'text' not in split or 'text_bert' not in split:
            generated, triples = encode(split['raw_text'], tokenizer, model, device)
            split.setdefault('text', generated)
            split.setdefault('text_bert', triples)
        out = {'test': split} if wrapped else split
        with open(destination / path.name, 'wb') as handle:
            pickle.dump(out, handle, protocol=4)
        print(f'{path.name}: {len(split["raw_text"])} sample(s)')
    print(f'Prepared {len(files)} files in {destination}')


if __name__ == '__main__':
    main()
