"""Data pipeline for the missing-aware, interpretable MOSEI competition model.

Attachment 3 marks missing information as *contiguous runs of all-zero
feature rows*.  Everything here therefore uses one rule for train, validation
and inference:

    real position     = position inside the sample's valid length
    observed position = real position whose feature row is not all zero

Training randomly zeros contiguous spans (the same corruption as Attachment 3),
validation is scored both on complete data and on a fixed, seeded contiguous
missing version, and features are standardized with training-set statistics
computed only over observed rows.
"""
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MODALITIES = ('text', 'audio', 'vision')
UNK_TOKEN_ID = 100


def normalize_classification_labels(labels, regression):
    """Map Negative/Neutral/Positive (strings, -1/0/1 or 0/1/2) to 0/1/2 and
    check that every label agrees with the sign of the regression label."""
    labels = np.asarray(labels).reshape(-1)
    regression = np.asarray(regression).reshape(-1)
    if labels.dtype.kind in {'U', 'S', 'O'}:
        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        normalized = np.asarray([mapping[str(v).strip().lower()] for v in labels], dtype=np.int64)
    else:
        normalized = labels.astype(np.int64)
        values = set(np.unique(normalized).tolist())
        if values.issubset({-1, 0, 1}):
            normalized = normalized + 1
        elif not values.issubset({0, 1, 2}):
            raise ValueError(f'Unexpected classification labels {sorted(values)}')
    expected = np.where(regression < 0, 0, np.where(regression > 0, 2, 1))
    mismatch = int(np.count_nonzero(normalized != expected))
    if mismatch:
        raise ValueError(f'{mismatch} classification labels disagree with regression_labels')
    return normalized


def clean_features(features):
    return np.nan_to_num(
        np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def infer_lengths(features):
    """Length = last non-zero row + 1 (used when no explicit length is stored)."""
    nonzero = np.any(features != 0, axis=-1)
    last = features.shape[1] - np.argmax(nonzero[:, ::-1], axis=1)
    return np.where(nonzero.any(axis=1), last, 1).astype(np.int64)


def text_tokens_observed(text_bert, real):
    return real & (text_bert[0] != UNK_TOKEN_ID)


def load_split(split, text_source):
    """Convert one pkl split (dict of field arrays) into model inputs."""
    arrays = {
        'audio': clean_features(split['audio']),
        'vision': clean_features(split['vision']),
    }
    has_text_bert = 'text_bert' in split
    if has_text_bert:
        arrays['text_bert'] = np.asarray(split['text_bert'], dtype=np.int64)
    if text_source == 'precomputed':
        if 'text' not in split:
            raise ValueError("text_source='precomputed' requires the 50x768 text field")
        arrays['text'] = clean_features(split['text'])
    elif not has_text_bert:
        raise ValueError("text_source='bert' requires the text_bert field")

    count = arrays['audio'].shape[0]
    lengths = {}
    for modality in ('audio', 'vision'):
        max_len = arrays[modality].shape[1]
        key = f'{modality}_lengths'
        if key in split:
            lengths[modality] = np.clip(np.asarray(split[key]).reshape(-1), 1, max_len)
        else:
            lengths[modality] = infer_lengths(arrays[modality])
    if has_text_bert:
        lengths['text'] = np.maximum(arrays['text_bert'][:, 1].sum(axis=1), 1)
    else:
        lengths['text'] = infer_lengths(arrays['text'])
    lengths = {key: value.astype(np.int64) for key, value in lengths.items()}

    record = {
        'arrays': arrays,
        'lengths': lengths,
        'ids': [str(value) for value in np.asarray(split.get('id', np.arange(count))).reshape(-1)],
        'raw_text': [str(value) for value in np.asarray(split.get('raw_text', [''] * count)).reshape(-1)],
        'count': count,
    }
    if 'regression_labels' in split:
        regression = np.asarray(split['regression_labels'], dtype=np.float32).reshape(-1)
        if 'classification_labels' in split:
            classes = normalize_classification_labels(
                split['classification_labels'], regression
            )
        else:
            classes = np.where(regression < 0, 0, np.where(regression > 0, 2, 1))
        record['regression'] = regression
        record['classes'] = classes.astype(np.int64)
    return record


def load_pickle(path, text_source):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return {name: load_split(data[name], text_source) for name in ('train', 'valid', 'test')}


def sample_spans(rng, length, min_ratio, max_ratio, max_spans):
    """Random contiguous spans covering roughly ratio * length positions."""
    ratio = rng.uniform(min_ratio, max_ratio)
    total = max(1, int(round(ratio * length)))
    n_spans = int(rng.integers(1, max_spans + 1))
    sizes = np.maximum(1, np.diff(np.sort(np.concatenate(
        [[0, total], rng.integers(0, total + 1, size=n_spans - 1)]
    ))))
    spans = []
    for size in sizes:
        size = int(min(size, length))
        start = int(rng.integers(0, length - size + 1))
        spans.append((start, start + size))
    return spans


def positional_spans(length, ratio, position):
    """Deterministic span for the missing-pattern analysis (begin/middle/end)."""
    size = max(1, int(round(ratio * length)))
    if position == 'begin':
        start = 0
    elif position == 'end':
        start = length - size
    elif position == 'middle':
        start = (length - size) // 2
    else:
        raise ValueError(f'Unknown missing position: {position}')
    return [(start, start + size)]


class FeatureNormalizer:
    """Per-dimension z-score using observed training rows only."""

    def __init__(self, stats=None, clip=10.0):
        self.stats = stats or {}
        self.clip = clip

    def fit(self, record, modalities):
        for modality in modalities:
            features = record['arrays'][modality]
            real = np.arange(features.shape[1])[None, :] < record['lengths'][modality][:, None]
            observed = real & np.any(features != 0, axis=-1)
            rows = features[observed]
            mean = rows.mean(axis=0)
            std = rows.std(axis=0)
            std[std < 1e-6] = 1.0
            self.stats[modality] = (mean.astype(np.float32), std.astype(np.float32))
        return self

    def transform(self, modality, features, observed):
        if modality not in self.stats:
            return features * observed[:, None]
        mean, std = self.stats[modality]
        normalized = np.clip((features - mean) / std, -self.clip, self.clip)
        return (normalized * observed[:, None]).astype(np.float32)

    def state_dict(self):
        return {'clip': self.clip, 'stats': {
            key: (mean.tolist(), std.tolist()) for key, (mean, std) in self.stats.items()
        }}

    @classmethod
    def from_state_dict(cls, state):
        stats = {
            key: (np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32))
            for key, (mean, std) in state['stats'].items()
        }
        return cls(stats=stats, clip=state['clip'])


def build_sample(raw, lengths, normalizer, text_source, spans=None):
    """Apply missing spans, derive masks and normalize one sample.

    raw: dict with 'audio', 'vision' and 'text' or 'text_bert' arrays.
    spans: optional {modality: [(start, end), ...]} zeroed before masking,
           exactly like the Attachment 3 corruption.
    """
    spans = spans or {}
    sample = {}
    for modality in MODALITIES:
        if modality == 'text' and text_source == 'bert':
            text_bert = raw['text_bert'].copy()
            max_len = text_bert.shape[1]
            for start, end in spans.get('text', []):
                # Keep [CLS] untouched; replace dropped word pieces with [UNK].
                start = max(start, 1)
                text_bert[0, start:end] = np.where(
                    text_bert[1, start:end] > 0, UNK_TOKEN_ID, text_bert[0, start:end]
                )
            real = np.arange(max_len) < lengths['text']
            sample['text_bert'] = torch.from_numpy(text_bert)
            sample['text_real'] = torch.from_numpy(real)
            sample['text_observed'] = torch.from_numpy(text_tokens_observed(text_bert, real))
            continue

        features = raw[modality].copy()
        for start, end in spans.get(modality, []):
            features[start:end] = 0.0
        real = np.arange(features.shape[0]) < lengths[modality]
        observed = real & np.any(features != 0, axis=-1)
        sample[modality] = torch.from_numpy(normalizer.transform(modality, features, observed))
        sample[f'{modality}_real'] = torch.from_numpy(real)
        sample[f'{modality}_observed'] = torch.from_numpy(observed)
    return sample


class RobustMSADataset(Dataset):
    """mode: 'train' (random corruption), 'complete', or 'missing' (fixed seed)."""

    def __init__(self, record, normalizer, cfg, mode='complete', seed=0, fixed_spans=None):
        self.record = record
        self.normalizer = normalizer
        self.text_source = cfg['text_source']
        self.aug = cfg['augment']
        # The fixed "missing" evaluation view must not change when an ablation
        # alters the training augmentation, so it has its own settings.
        self.eval_aug = cfg.get('eval_missing', cfg['augment'])
        self.mode = mode
        self.fixed_spans = fixed_spans
        if mode == 'missing' and fixed_spans is None:
            rng = np.random.default_rng(seed)
            self.fixed_spans = [self._random_spans(rng, index, self.eval_aug)
                                for index in range(record['count'])]

    def _length(self, modality, index):
        return int(self.record['lengths'][modality][index])

    def _random_spans(self, rng, index, aug=None):
        aug = aug or self.aug
        spans = {}
        for modality in MODALITIES:
            if rng.random() < aug['span_prob']:
                spans[modality] = sample_spans(
                    rng, self._length(modality, index),
                    aug['min_ratio'], aug['max_ratio'], aug['max_spans'],
                )
        if rng.random() < aug['modality_drop_prob']:
            # Drop one whole modality, never all three.
            modality = MODALITIES[int(rng.integers(0, 3))]
            spans[modality] = [(0, self._length(modality, index))]
        return spans

    def __len__(self):
        return self.record['count']

    def __getitem__(self, index):
        arrays = self.record['arrays']
        raw = {key: value[index] for key, value in arrays.items()}
        lengths = {key: int(value[index]) for key, value in self.record['lengths'].items()}
        if self.mode == 'train':
            spans = self._random_spans(np.random.default_rng(), index)
        elif self.mode == 'missing':
            spans = self.fixed_spans[index]
        else:
            spans = None
        sample = build_sample(raw, lengths, self.normalizer, self.text_source, spans)
        sample['index'] = index
        if 'regression' in self.record:
            sample['regression'] = torch.tensor(self.record['regression'][index])
            sample['classes'] = torch.tensor(self.record['classes'][index])
        return sample


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=False,
        persistent_workers=num_workers > 0,
    )
