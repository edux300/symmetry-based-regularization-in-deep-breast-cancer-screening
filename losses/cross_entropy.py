from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch

def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss



#  Implementation of Label smoothing with CrossEntropy and ignore_index
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean',ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
        return linear_combination(loss/n, nll, self.epsilon)


# Implementation of Label smoothing with NLLLoss and ignore_index
class LabelSmoothingNLLLoss(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean',ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self, preds, target):
        n = preds.size()[-1]
        loss = reduce_loss(-preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
        return linear_combination(loss/n, nll, self.epsilon)
    
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0, specific_class=None):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.specific_class = specific_class
            

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        if self.specific_class is not None:
            idxs = targets != self.specific_class
            final_targets = torch.empty(size=(targets.size(0), inputs.size(-1)), device=targets.device)
            final_targets[idxs] = self.k_one_hot(targets, inputs.size(-1), self.smoothing)[idxs]
            final_targets[torch.logical_not(idxs)] = self.k_one_hot(targets, inputs.size(-1), 0)[torch.logical_not(idxs)]
            targets = final_targets
        else:
            targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)

        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))

