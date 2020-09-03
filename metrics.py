import torch

def accuracy(output, target, threshold=0.5):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = torch.ge(output, threshold).float()

        true_positive = torch.sum(output * target)
        true_negative = torch.sum((1 - output) * (1 - target))

        return (true_positive + true_negative) / target.size(0)


# def accuracy(output, target):
#     with torch.no_grad():
#         _, output = torch.max(output, 1)
#         _, target = torch.max(target, 1)
#         return (output == target).sum().float() / target.size(0)


def fbeta_score(output, target, beta=1, threshold=0.5, eps=1e-9):
    with torch.no_grad():
        beta = beta ** 2

        output = torch.sigmoid(output)
        output = torch.ge(output, threshold).float()

        true_positive = torch.sum(output * target)

        precision = true_positive / (output.sum() + eps)
        recall = true_positive / (target.sum() + eps)

        return (1 + beta) * precision * recall / (beta * precision + recall + eps)
