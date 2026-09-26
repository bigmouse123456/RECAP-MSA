"""Local perturbation explanations for RobustMSA predictions.

Gate and temporal attention are reported as model diagnostics. The signed
occlusion delta below is a direct, repeatable prediction change. It is a
counterfactual for this trained model, not a causal claim about the speaker.
"""
import numpy as np
import torch

MODALITIES = ('text', 'audio', 'vision')


def ablate(batch, modality, start=None, end=None):
    changed = dict(batch)
    observed = batch[f'{modality}_observed'].clone()
    if start is None:
        observed[:] = False
        start, end = 0, observed.shape[1]
    else:
        observed[:, start:end] = False
    changed[f'{modality}_observed'] = observed
    if modality == 'text' and 'text_bert' in batch:
        text_bert = batch['text_bert'].clone()
        # The first channel holds token IDs. Keep attention mask and segment IDs.
        text_bert[:, 0, start:end] = 100
        changed['text_bert'] = text_bert
    else:
        values = batch[modality].clone()
        values[:, start:end] = 0
        changed[modality] = values
    return changed


@torch.no_grad()
def occlusion_delta(model, batch, baseline, modality, start=None, end=None):
    masked = model(ablate(batch, modality, start, end))
    return float((baseline['intensity'] - masked['intensity']).item())


@torch.no_grad()
def explain_ensemble(runs, top_k=3):
    """runs: list of (model, sample_batch, model_output) for one sample."""
    weights = {modality: [] for modality in MODALITIES}
    for model, batch, output in runs:
        for modality in MODALITIES:
            weights[modality].append(output['temporal_weights'][modality][0].cpu().numpy())

    explanation = {}
    for modality in MODALITIES:
        whole = [occlusion_delta(model, batch, output, modality)
                 for model, batch, output in runs]
        explanation[f'{modality}_removal_delta'] = float(np.mean(whole))
        attention = np.mean(weights[modality], axis=0)
        candidates = np.argsort(-attention)[:max(0, top_k)]
        evidence = []
        for step in candidates:
            if attention[step] <= 0:
                continue
            start = int(step) * runs[0][0].strides[modality]
            end = min(start + runs[0][0].strides[modality],
                      runs[0][1][f'{modality}_observed'].shape[1])
            if not runs[0][1][f'{modality}_observed'][0, start:end].any():
                continue
            delta = np.mean([occlusion_delta(model, batch, output, modality, start, end)
                             for model, batch, output in runs])
            evidence.append((start, end, float(attention[step]), float(delta)))
        explanation[f'{modality}_windows'] = evidence
    return explanation
