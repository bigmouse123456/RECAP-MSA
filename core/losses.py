from torch import nn
from torch.nn import functional as F
import torch

class MultimodalLoss_stage1(nn.Module):
    def __init__(self, args):
        super().__init__()
        # adv loss, SSCL loss, recon loss
        self.adv = args['base']['adv']
        self.recon = args['base']['recon']
        self.paraSSL1 = args['base']['paraSSL_LV']
        self.paraSSL2 = args['base']['paraSSL_LA']
        self.paraSSL3 = args['base']['paraSSL_VA']
        self.MSE_Fn = nn.MSELoss() 


    def forward(self, out, label):
        # adv loss, SSCL loss, recon loss
        l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out['complete_feats'] is not None else 0

        l_ssl_LV = max(out['Factorized_LV'], -5.0)
        l_ssl_LA = max(out['Factorized_LA'], -5.0)
        l_ssl_VA = max(out['Factorized_VA'], -5.0)

        l_adv = max(out['loss_adv'], -3.0)
        loss = self.recon * l_rec  + self.adv * l_adv + self.paraSSL1 * l_ssl_LV + self.paraSSL2 * l_ssl_LA + self.paraSSL3 * l_ssl_VA

        return {'loss': loss, 'l_rec': l_rec, 'l_adv': l_adv, 'l_ssl_LV': l_ssl_LV, 'l_ssl_LA': l_ssl_LA, 'l_ssl_VA': l_ssl_VA}


class MultimodalLoss_stage2(nn.Module):
    def __init__(self, args, class_weights=None):
        super().__init__()
        # fusion, prediction loss
        self.task = args['base'].get('task_reg', args['base']['task'])
        self.task_cls = args['base'].get('task_cls', 0.4)
        self.task_modal = args['base'].get('task_modal', 0.1)
        self.task_corr = args['base'].get('task_corr', 0.1)
        self.task_order = args['base'].get('task_order', 0.05)
        self.task_consistency = args['base'].get('task_consistency', 0.05)
        self.mae_mix = args['base'].get('mae_mix', 0.5)
        self.para_rank = args['base']['para_rank']
        self.regression_fn = nn.SmoothL1Loss()
        self.classification_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args['base'].get('label_smoothing', 0.05),
        )

    @staticmethod
    def concordance_loss(predictions, targets, eps=1e-8):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        pred_mean = predictions.mean()
        target_mean = targets.mean()
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        covariance = (pred_centered * target_centered).mean()
        pred_variance = pred_centered.square().mean()
        target_variance = target_centered.square().mean()
        ccc = (2.0 * covariance) / (
            pred_variance
            + target_variance
            + (pred_mean - target_mean).square()
            + eps
        )
        return 1.0 - ccc

    @staticmethod
    def pairwise_order_loss(predictions, targets, min_difference=0.05):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        target_difference = targets[:, None] - targets[None, :]
        prediction_difference = predictions[:, None] - predictions[None, :]
        valid = torch.triu(
            target_difference.abs(), diagonal=1
        ) > min_difference
        if not valid.any():
            return predictions.new_zeros(())
        direction = target_difference[valid].sign()
        return F.softplus(-direction * prediction_difference[valid]).mean()


    def forward(self, out, label):
        predictions = out['sentiment_preds']
        targets = label['sentiment_labels']
        l_huber = self.regression_fn(predictions, targets)
        l_mae = F.l1_loss(predictions, targets)
        l_sp = (1.0 - self.mae_mix) * l_huber + self.mae_mix * l_mae
        l_corr = self.concordance_loss(predictions, targets)
        l_order = self.pairwise_order_loss(predictions, targets)
        l_cls = self.classification_fn(
            out['polarity_logits'], label['classification_labels'].view(-1).long()
        )

        l_ranking = out['ranking_loss']
        modal_targets = label['sentiment_labels'].unsqueeze(1).expand_as(
            out['modal_predictions']
        )
        l_modal = self.regression_fn(out['modal_predictions'], modal_targets)

        polarity_probabilities = F.softmax(out['polarity_logits'], dim=-1)
        polarity_scale = predictions.new_tensor([-1.0, 0.0, 1.0])
        expected_polarity = (
            polarity_probabilities * polarity_scale.unsqueeze(0)
        ).sum(dim=-1, keepdim=True)
        l_consistency = F.smooth_l1_loss(expected_polarity, predictions / 3.0)

        loss = (
            self.task * l_sp
            + self.task_cls * l_cls
            + self.task_modal * l_modal
            + self.task_corr * l_corr
            + self.task_order * l_order
            + self.task_consistency * l_consistency
            + self.para_rank * l_ranking
        )

        return {
            'loss': loss,
            'l_sp': l_sp,
            'l_mae': l_mae,
            'l_corr': l_corr,
            'l_order': l_order,
            'l_cls': l_cls,
            'l_modal': l_modal,
            'l_consistency': l_consistency,
            'ranking': l_ranking,
        }
