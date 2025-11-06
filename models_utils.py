import torch
import torch.nn as nn
import torch.nn.functional as F

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

class HeatmapHead(nn.Module):
    def __init__(self, in_channels, num_points, hm_h=64, hm_w=64):
        super().__init__()
        self.hm_h = hm_h
        self.hm_w = hm_w
        self.num_points = num_points

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_points, kernel_size=1)
        )

        nn.init.normal_(self.head[-1].weight, std=0.001)
        if self.head[-1].bias is not None:
            nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, x):
        out = self.head(x)
        out = F.interpolate(out, size=(self.hm_h, self.hm_w), mode='bilinear', align_corners=False)
        return out
