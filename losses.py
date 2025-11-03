import torch
import torch.nn as nn

class WingLoss(nn.Module):
    def __init__(self, omega=10.0, epsilon=2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
        self.register_buffer("C", C)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        mask = (diff < self.omega).float()
        loss = mask * (self.omega * torch.log1p(diff / self.epsilon)) + (1.0 - mask) * (diff - self.C)
        return loss.mean()


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        case1 = diff < self.theta
        A = self.omega * (1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target))) \
            * (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1.0) * (1.0 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        loss = torch.where(case1,
                           self.omega * torch.log1p(torch.pow(diff / self.epsilon, self.alpha - target)),
                           A * diff - C)
        return loss.mean()


class HeatmapFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred_logits, gt):
        p = torch.sigmoid(pred_logits)
        pos_mask = (gt > 0.0).float()
        neg_mask = (1.0 - pos_mask)

        pos_loss = - ( (1 - p) ** self.alpha ) * torch.log(p + self.eps) * pos_mask
        neg_loss = - ( (1 - gt) ** self.beta ) * ( p ** self.alpha ) * torch.log(1 - p + self.eps) * neg_mask

        loss = pos_loss.sum() + neg_loss.sum()
        n_pos = pos_mask.sum()
        if n_pos > 0:
            loss = loss / n_pos
        else:
            loss = loss / (pred_logits.shape[0])
        return loss