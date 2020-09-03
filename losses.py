import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p = torch.exp(-bce)
        loss = self.alpha * (1 - p) ** self.gamma * bce

        if self.reduce:
            loss = torch.mean(loss)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True, **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce)
        loss = self.alpha * (1 - p) ** self.gamma * ce

        if self.reduce:
            return torch.mean(loss)
        return loss


class BCELoss(nn.Module):
    def __init__(self, reduce=True):
        super(BCELoss, self).__init__()
        self.reduce = reduce

    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        if self.reduce:
            loss = torch.mean(loss)

        return loss


class CELoss(nn.Module):
    def __init__(self, reduce=True):
        super(CELoss, self).__init__()
        self.reduce = reduce

    def forward(self, inputs, targets):
        loss = F.cross_entropy(inputs, targets, reduction="none")

        if self.reduce:
            loss = torch.mean(loss)

        return loss


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05, reduce=True):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.reduce = reduce

    def forward(self, inputs, targets):
        logprobs = F.log_softmax(inputs, dim=-1)

        nll_loss = -logprobs * targets
        nll_loss = torch.sum(nll_loss, dim=-1)

        smooth_loss = -torch.mean(logprobs, dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduce:
            loss = torch.mean(loss)

        return loss


def focal(*argv, **kwargs):
    return FocalLoss(*argv, **kwargs)


def ce(*argv, **kwargs):
    return CELoss(*argv, **kwargs)


def binary_ce(*argv, **kwargs):
    return BCELoss(*argv, **kwargs)


def binary_focal(*argv, **kwargs):
    return BinaryFocalLoss(*argv, **kwargs)


def label_smoothing(*argv, **kwargs):
    return LabelSmoothing(*argv, **kwargs)
