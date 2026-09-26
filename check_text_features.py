"""Check whether Attachment 2's 50x768 `text` field can be regenerated from raw_text.

Attachments 3/4 only ship raw_text for the text modality, so the precomputed
text features must be rebuilt at inference time.  This script verifies, on
Attachment 2 samples, that (1) tokenizing raw_text reproduces `text_bert`, and
(2) which BERT hidden layer reproduces the stored `text` features.

    python check_text_features.py --data data/mosei/unaligned_50.pkl
"""
import argparse
import pickle

import numpy as np
import torch
from transformers import BertModel, BertTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/mosei/unaligned_50.pkl')
    parser.add_argument('--split', default='valid')
    parser.add_argument('--pretrained', default='bert-base-uncased')
    parser.add_argument('--n', type=int, default=64)
    args = parser.parse_args()

    with open(args.data, 'rb') as handle:
        data = pickle.load(handle)[args.split]
    n = args.n
    text = np.asarray(data['text'][:n], dtype=np.float32)
    text_bert = np.asarray(data['text_bert'][:n]).astype(np.int64)
    raw = [str(value) for value in np.asarray(data['raw_text'][:n]).reshape(-1)]
    mask = text_bert[:, 1].astype(bool)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    encoded = tokenizer(raw, max_length=text_bert.shape[-1], padding='max_length',
                        truncation=True, return_token_type_ids=True)
    ids = np.asarray(encoded['input_ids'])
    same = (ids == text_bert[:, 0]).all(axis=1)
    print(f'raw_text -> text_bert ids identical: {same.mean():.1%} of {n} samples')
    if not same.all():
        index = int(np.argmin(same))
        print('  first mismatch raw_text:', raw[index])
        print('  stored :', tokenizer.convert_ids_to_tokens(text_bert[index, 0][mask[index]].tolist()))
        print('  retoken:', tokenizer.convert_ids_to_tokens(ids[index][ids[index] != 0].tolist()))

    padded = text[~mask]
    print(f'stored text at padding positions: mean |x| = {np.abs(padded).mean():.4f}')

    model = BertModel.from_pretrained(args.pretrained, output_hidden_states=True).eval()
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(text_bert[:, 0]),
                    attention_mask=torch.from_numpy(text_bert[:, 1]),
                    token_type_ids=torch.from_numpy(text_bert[:, 2]))
    target = text[mask]
    results = []
    for layer, hidden in enumerate(out.hidden_states):
        pred = hidden.numpy()[mask]
        cos = (pred * target).sum(-1) / (
            np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1) + 1e-8)
        results.append((float(cos.mean()), float(np.abs(pred - target).max()), layer))
        print(f'layer {layer:2d}: mean cosine {cos.mean():.4f}, max |diff| {np.abs(pred - target).max():.4f}')
    best = max(results)
    print(f'BEST layer {best[2]} (cosine {best[0]:.4f}); '
          f'use --text_layer {best[2] - len(out.hidden_states)} in predict_robust.py '
          f'if cosine > 0.99')


if __name__ == '__main__':
    main()
