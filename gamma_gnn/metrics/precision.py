"""
Binary accuracy metric class for tpc_ml.
"""
import torch
import torch.nn as nn
from sklearn.metrics import precision_score

from gamma_gnn.metrics import GenericMetric

class Precision(GenericMetric):
    
    def __init__(self,
        name:   str='binary_accuracy',
        output_shape:   tuple=(),
        target_shape:    tuple=(),
        cutoff:         float=0.5,
    ):
        """
        Binary accuracy metric which essentially computes
        the number of correct guesses defined by a single
        cut along the output dimension.
        """
        super(Precision, self).__init__(
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
        precision = precision_score(predictions, data.y.cpu().detach().numpy())
        self.batch_metric = torch.cat(
            (self.batch_metric, torch.tensor([[precision]], device=self.device)), 
            dim=0
        )

    def compute(self):
        return self.batch_metric.mean()