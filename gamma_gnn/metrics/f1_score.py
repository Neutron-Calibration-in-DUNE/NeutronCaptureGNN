"""
Binary accuracy metric class for tpc_ml.
"""
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from gamma_gnn.metrics import GenericMetric

class F1Score(GenericMetric):
    
    def __init__(self,
        name:   str='f1_score',
        output_shape:   tuple=(),
        target_shape:    tuple=(),
        cutoff:         float=0.5,
    ):
        """
        """
        super(F1Score, self).__init__(
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
        f1score = f1_score(predictions, data.y.cpu().detach().numpy())
        self.batch_metric = torch.cat(
            (self.batch_metric, torch.tensor([[f1score]], device=self.device)), 
            dim=0
        )

    def compute(self):
        return self.batch_metric.mean()