"""
Binary accuracy metric class for tpc_ml.
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from gamma_gnn.metrics import GenericMetric

class ROCAUC(GenericMetric):
    
    def __init__(self,
        name:   str='roc_auc',
        output_shape:   tuple=(),
        target_shape:    tuple=(),
        cutoff:         float=0.5,
    ):
        """
        """
        super(ROCAUC, self).__init__(
            name,
            output_shape,
            target_shape,
        )
        self.cutoff = cutoff

    def update(self,
        outputs,
        data,
    ):
        # set predictions using cutoff
        predictions = (outputs.squeeze(1) > self.cutoff).unsqueeze(1).cpu().detach().numpy()
        try:
            roc_auc = roc_auc_score(predictions, data.y.cpu().detach().numpy())
        except:
            roc_auc = 0.0
        self.batch_metric = torch.cat(
            (self.batch_metric, torch.tensor([[roc_auc]], device=self.device)), 
            dim=0
        )

    def compute(self):
        return self.batch_metric.mean()