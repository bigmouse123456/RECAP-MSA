"""Metrics, polarity decision rules and batched inference for RobustMSA."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

POLARITY_NAMES = ('Negative', 'Neutral', 'Positive')
DEFAULT_DECISION = {'type': 'logit_bias', 'bias': [0.0, 0.0, 0.0]}


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def collect(model, loader, device):
    """Run the model over a loader and gather numpy outputs."""
    model.eval()
    keys = ('intensity', 'probs', 'modality_weights', 'unimodal_preds', 'reliability')
    out = {key: [] for key in keys}
    out.update({'regression': [], 'classes': [], 'index': []})
    for batch in loader:
        batch = to_device(batch, device)
        result = model(batch)
        out['intensity'].append(result['intensity'].cpu())
        out['probs'].append(torch.softmax(result['polarity_logits'], dim=-1).cpu())
        for key in ('modality_weights', 'unimodal_preds', 'reliability'):
            out[key].append(result[key].cpu())
        out['index'].append(batch['index'].cpu())
        if 'regression' in batch:
            out['regression'].append(batch['regression'].cpu())
            out['classes'].append(batch['classes'].cpu())
    return {key: torch.cat(value).numpy() for key, value in out.items() if value}


def decide(probs, intensity, decision):
    if decision['type'] == 'thresholds':
        low, high = decision['thresholds']
        return np.where(intensity < low, 0, np.where(intensity > high, 2, 1))
    return np.argmax(np.log(probs + 1e-9) + np.asarray(decision['bias']), axis=-1)


def macro_f1(truth, pred):
    return f1_score(truth, pred, average='macro', zero_division=0)


def calibrate_decision(probs, intensity, classes):
    """Choose the polarity rule on validation data only (allowed by the rules).

    Candidates: classifier argmax with additive class log-biases, or
    thresholds on the predicted intensity.  Returns the best by Macro-F1.
    """
    best = (macro_f1(classes, decide(probs, intensity, DEFAULT_DECISION)), DEFAULT_DECISION)
    log_probs = np.log(probs + 1e-9)
    grid = np.round(np.arange(-2.0, 2.01, 0.1), 2)
    for neutral in grid:
        for positive in grid:
            bias = [0.0, float(neutral), float(positive)]
            score = macro_f1(classes, np.argmax(log_probs + np.asarray(bias), axis=-1))
            if score > best[0] + 1e-6:
                best = (score, {'type': 'logit_bias', 'bias': bias})
    for low in np.round(np.arange(-1.0, 0.001, 0.05), 2):
        for high in np.round(np.arange(0.0, 1.001, 0.05), 2):
            decision = {'type': 'thresholds', 'thresholds': [float(low), float(high)]}
            score = macro_f1(classes, decide(probs, intensity, decision))
            if score > best[0] + 1e-6:
                best = (score, decision)
    return best[1], best[0]


def compute_metrics(outputs, decision=DEFAULT_DECISION):
    pred = outputs['intensity']
    truth = outputs['regression']
    classes = outputs['classes']
    polarity = decide(outputs['probs'], pred, decision)
    non_zero = truth != 0
    # Near-constant predictions (e.g. every modality missing) have no Pearson r.
    corr = float(np.corrcoef(pred, truth)[0, 1]) if pred.std() > 1e-6 else 0.0
    return {
        'MAE': round(float(np.abs(pred - truth).mean()), 4),
        'Corr': round(corr, 4),
        'Polarity_Accuracy': round(float(accuracy_score(classes, polarity)), 4),
        'Polarity_Macro_F1': round(float(macro_f1(classes, polarity)), 4),
        'Non0_acc_2': round(float(np.mean((pred[non_zero] > 0) == (truth[non_zero] > 0))), 4),
        'Mult_acc_7': round(float(np.mean(
            np.round(np.clip(pred, -3, 3)) == np.round(np.clip(truth, -3, 3))
        )), 4),
    }


def selection_score(metrics):
    return metrics['Polarity_Macro_F1'] + metrics['Corr'] - metrics['MAE']


def merge_outputs(outputs_list):
    keys = set.intersection(*(set(output) for output in outputs_list))
    return {key: np.concatenate([output[key] for output in outputs_list]) for key in keys}


def average_outputs(outputs_list):
    """Ensemble several checkpoints evaluated on the same loader."""
    merged = dict(outputs_list[0])
    for key in ('intensity', 'probs', 'modality_weights', 'unimodal_preds'):
        merged[key] = np.mean([output[key] for output in outputs_list], axis=0)
    return merged
