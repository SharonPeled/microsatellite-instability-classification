# copied from https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
import torch
import torch.nn.functional as F


class BSCELoss(torch.nn.Module):
    "Binary Symmetric Cross Entropy loss"
    def __init__(self, alpha, beta):
        super(BSCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, labels, pos_weight):
        return BSCELoss.functional(pred, labels, pos_weight, self.alpha, self.beta)

    @staticmethod
    def functional(pred, labels, pos_weight, alpha, beta):
        # CCE
        ce = F.binary_cross_entropy_with_logits(pred, labels, reduction='mean',
                                                pos_weight=pos_weight)
        # RCE
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        rce = F.binary_cross_entropy_with_logits(torch.clamp(labels, min=1e-4, max=1.0),
                                                 pred, reduction='none')
        # Manual implementation of pos_weight
        rce_weighted_losses = rce * (labels * (pos_weight - 1) + 1)
        rce = rce_weighted_losses.mean()
        # Loss
        loss = alpha * ce + beta * rce
        return loss
