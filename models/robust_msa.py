"""Missing-aware, interpretable multimodal sentiment model (RobustMSA).

Each modality is encoded by a small masked Transformer.  Missing positions
(all-zero rows) are replaced by a learned "missing" embedding so the encoder
can use the surrounding context, but they are excluded from the temporal
attention pooling: a missing frame can never be reported as key evidence.

The per-modality summaries are fused by a reliability-aware gate (the share of
observed positions is an explicit gate input), so a heavily corrupted modality
is down-weighted instead of injecting noise.  Outputs used for the competition:

    intensity           continuous score in [-3, 3]
    polarity_logits     Negative / Neutral / Positive
    modality_weights    per-sample contribution of text / audio / vision
    temporal_weights    per-position evidence weights inside each modality
    unimodal_preds      what each modality alone would predict
"""
import torch
import torch.nn.functional as F
from torch import nn

MODALITIES = ('text', 'audio', 'vision')


def masked_softmax(scores, mask):
    scores = scores.masked_fill(~mask, -1e4)
    weights = torch.softmax(scores, dim=-1) * mask
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def masked_downsample(x, real, observed, stride):
    """Average non-overlapping windows over observed rows only."""
    if stride <= 1:
        return x, real, observed
    batch, length, dim = x.shape
    pad = (-length) % stride
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        real = F.pad(real.float(), (0, pad)).bool()
        observed = F.pad(observed.float(), (0, pad)).bool()
    steps = x.shape[1] // stride
    x = x.view(batch, steps, stride, dim)
    obs = observed.view(batch, steps, stride)
    total = (x * obs.unsqueeze(-1)).sum(dim=2)
    count = obs.sum(dim=2, keepdim=True).clamp_min(1)
    return total / count, real.view(batch, steps, stride).any(dim=2), obs.any(dim=2)


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, max_len, cfg):
        super().__init__()
        dim = cfg['hidden_dim']
        self.stride = 1
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(cfg['input_dropout']),
            nn.Linear(input_dim, dim),
        )
        self.position = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.missing_token = nn.Parameter(torch.zeros(dim))
        layer = nn.TransformerEncoderLayer(
            dim, cfg['heads'], dim * 2, cfg['dropout'],
            batch_first=True, norm_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg['layers'], enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x, real, observed):
        h = self.input(x)
        h = torch.where(observed.unsqueeze(-1), h, self.missing_token.expand_as(h))
        h = h + self.position[:, :h.size(1)]
        # Every row needs at least one attendable key to avoid NaNs.
        attendable = real.clone()
        attendable[:, 0] = True
        h = self.norm(self.encoder(h, src_key_padding_mask=~attendable))
        weights = masked_softmax(self.pool(h).squeeze(-1), observed)
        pooled = torch.einsum('bt,btd->bd', weights, h)
        return pooled, weights


class TextBertFrontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from models.bert import BertTextEncoder
        self.bert = BertTextEncoder(use_finetune=True, pretrained=cfg['bert_pretrained'])
        frozen = cfg.get('bert_frozen_layers', 8)
        for parameter in self.bert.model.embeddings.parameters():
            parameter.requires_grad = False
        for layer in self.bert.model.encoder.layer[:frozen]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(self, text_bert):
        return self.bert(text_bert.float())


class RobustMSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg['model']
        self.text_source = cfg['data']['text_source']
        dim = model_cfg['hidden_dim']
        self.text_frontend = TextBertFrontend(model_cfg) if self.text_source == 'bert' else None

        self.strides = {m: model_cfg['downsample'].get(m, 1) for m in MODALITIES}
        self.encoders = nn.ModuleDict()
        for modality in MODALITIES:
            max_len = -(-model_cfg['max_len'][modality] // self.strides[modality])
            self.encoders[modality] = ModalityEncoder(
                model_cfg['input_dims'][modality], max_len, model_cfg
            )

        # Gate input: pooled feature + observed ratio + availability flag.
        self.gate = nn.Sequential(nn.Linear(dim + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.modality_embedding = nn.Parameter(torch.zeros(3, dim))
        fusion_dropout = model_cfg['fusion_dropout']
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Dropout(fusion_dropout),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )
        self.regression_head = nn.Linear(dim, 1)
        self.polarity_head = nn.Linear(dim, 3)
        self.unimodal_heads = nn.ModuleList([nn.Linear(dim, 1) for _ in MODALITIES])

    def encode(self, batch):
        pooled, temporal, reliability, available = [], {}, [], []
        for modality in MODALITIES:
            real = batch[f'{modality}_real']
            observed = batch[f'{modality}_observed']
            if modality == 'text' and self.text_frontend is not None:
                x = self.text_frontend(batch['text_bert'])
            else:
                x = batch[modality]
            x, real, observed = masked_downsample(x, real, observed, self.strides[modality])
            vector, weights = self.encoders[modality](x, real, observed)
            ratio = observed.sum(dim=1).float() / real.sum(dim=1).clamp_min(1).float()
            has_any = observed.any(dim=1)
            pooled.append(vector * has_any.unsqueeze(-1))
            temporal[modality] = weights
            reliability.append(ratio)
            available.append(has_any)
        return (torch.stack(pooled, dim=1), temporal,
                torch.stack(reliability, dim=1), torch.stack(available, dim=1))

    def forward(self, batch):
        pooled, temporal, reliability, available = self.encode(batch)
        pooled = pooled + self.modality_embedding * available.unsqueeze(-1)

        gate_input = torch.cat(
            [pooled, reliability.unsqueeze(-1), available.float().unsqueeze(-1)], dim=-1
        )
        gate_scores = self.gate(gate_input).squeeze(-1)
        # If all three modalities are absent fall back to a uniform average.
        gate_mask = available | ~available.any(dim=1, keepdim=True)
        modality_weights = masked_softmax(gate_scores, gate_mask)
        fused = torch.einsum('bm,bmd->bd', modality_weights, pooled)

        hidden = self.fusion(torch.cat([fused, pooled.flatten(1)], dim=-1))
        intensity = 3.0 * torch.tanh(self.regression_head(hidden).squeeze(-1) / 3.0)
        unimodal = torch.cat(
            [head(pooled[:, i]) for i, head in enumerate(self.unimodal_heads)], dim=-1
        )
        return {
            'intensity': intensity,
            'polarity_logits': self.polarity_head(hidden),
            'modality_weights': modality_weights,
            'temporal_weights': temporal,
            'unimodal_preds': 3.0 * torch.tanh(unimodal / 3.0),
            'reliability': reliability,
            'available': available,
        }


class RobustMSALoss(nn.Module):
    def __init__(self, cfg, class_weights=None):
        super().__init__()
        loss_cfg = cfg['loss']
        self.w_cls = loss_cfg['cls']
        self.w_uni = loss_cfg['unimodal']
        self.w_corr = loss_cfg['corr']
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=loss_cfg['label_smoothing'])

    @staticmethod
    def pearson_loss(pred, target):
        pred = pred - pred.mean()
        target = target - target.mean()
        corr = (pred * target).sum() / (pred.norm() * target.norm() + 1e-8)
        return 1.0 - corr

    def forward(self, out, regression, classes):
        l_mae = F.l1_loss(out['intensity'], regression)
        l_cls = self.ce(out['polarity_logits'], classes)
        uni_mask = out['available'].float()
        uni_error = (out['unimodal_preds'] - regression.unsqueeze(-1)).abs() * uni_mask
        l_uni = uni_error.sum() / uni_mask.sum().clamp_min(1)
        l_corr = self.pearson_loss(out['intensity'], regression)
        loss = l_mae + self.w_cls * l_cls + self.w_uni * l_uni + self.w_corr * l_corr
        return loss, {'loss': loss.item(), 'mae': l_mae.item(), 'cls': l_cls.item(),
                      'uni': l_uni.item(), 'corr': l_corr.item()}
