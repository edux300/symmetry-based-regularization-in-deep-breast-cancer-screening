from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch
import math

class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 1-(p*z).sum(dim=1).mean()

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return 1-(z1 * z2).sum(dim=1).mean()

class Norm2SimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        return (((z1 - z2) **2 ).sum(dim=1) / 45.25).mean()


def kl_ab(a, b, eps=1e-8):
    return ((a+eps) * ((a+eps) / (b+eps)).log()).sum(dim=1).mean()


class RepresentationInvarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, p1, p2, T=5):
        p1 = F.softmax(p1/T, dim=1)
        p2 = F.softmax(p2/T, dim=1)
        m = (p1+p2) / 2
        return (kl_ab(p1, m) + kl_ab(p2, m)) * T ** 2


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, z1, z2, pos_margin=0, neg_margin=1):
        n, f = z1.shape
        z1 = z1.unsqueeze(1)
        z2 = z2.unsqueeze(0)
        matrix = ((z1-z2)**2).sum(dim=2) / math.sqrt(f)
        dpos = matrix.diag()
        dneg = matrix.fill_diagonal_(0)

        loss_pos = (torch.maximum(dpos-pos_margin, torch.zeros_like(dpos)).sum() / n)
        loss_neg = (torch.maximum(neg_margin-dneg, torch.zeros_like(dneg)).sum() / (n*(n-1)))
        return loss_pos + loss_neg
