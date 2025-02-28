from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module): 
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))


        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=mask_pos_pairs.device), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class ModifiedSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ModifiedSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # [batch_size, feature_dim]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [batch_size * n_views, feature_dim]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_feature = nn.functional.normalize(anchor_feature, dim=1)
        contrast_feature = nn.functional.normalize(contrast_feature, dim=1)

        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)  # [anchor_count * batch_size, contrast_count * batch_size]

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask  # [anchor_count * batch_size, contrast_count * batch_size]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [anchor_count * batch_size, contrast_count * batch_size]
        mask_pos_pairs = mask.sum(1)  # [anchor_count * batch_size]
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 
                                     torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=mask_pos_pairs.device), 
                                     mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs  # [anchor_count * batch_size]

        alignment = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # [anchor_count * batch_size]
        sum_neg = exp_logits.sum()  # Scalar
        uniformity = torch.log(2 * batch_size * sum_neg)  # Scalar

        loss_alignment = alignment.mean()
        loss_uniformity = uniformity

        loss = loss_alignment + loss_uniformity

        return loss
