"""
Wrapper for L2 loss
"""
import torch
import torch.nn as nn

from gamma_gnn.losses import GenericLoss

class BCELoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='bce_loss',
        reduction:  str='mean',
    ):
        super(BCELoss, self).__init__(name)
        self.alpha = alpha
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.bce_loss(outputs.squeeze(1), data.y.to(self.device).float())
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss