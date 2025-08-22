import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, config, gamma=3, reduction='mean'):
        """
        alpha: Tensor, shape [num_classes] 给每类一个权重
        gamma: 聚焦参数
        """
        self.config=config
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(config.crowd_weight)
        self.gamma = config.loss_gamma
        self.reduction = reduction

    def forward(self, logits, target):
        if self.config.crowd_label_type=="soft":
            return self.soft_forward(logits, target)
        else:
            return self.hard_forward(logits, target)

    def hard_forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none', weight=self.alpha.to(logits.device))
        pt = torch.exp(-ce_loss)  # pt = softmax(logit)[target]
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def soft_forward(self, logits, soft_target):

        log_probs = F.log_softmax(logits, dim=1)  # [B, C]

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)  # [C]
            # 给每个类别加权
            weighted_target = soft_target * alpha.unsqueeze(0)  # [B, C]
            weighted_target = weighted_target / weighted_target.sum(dim=1, keepdim=True)  # 归一化，确保概率和为1
        else:
            weighted_target = soft_target

        loss = -(weighted_target * log_probs).sum(dim=1)  # [B]

        if self.reduction == 'batchmean':
            return loss.mean()  # kl_div 的 batchmean 是对 batch 求均值
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
