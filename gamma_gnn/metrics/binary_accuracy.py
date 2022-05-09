"""
Binary accuracy metric class for tpc_ml.
"""
import torch
import torch.nn as nn

from gamma_gnn.metrics import GenericMetric

class BinaryAccuracy(GenericMetric):
    
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
        super(BinaryAccuracy, self).__init__(
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
        predictions = (outputs.squeeze(1) > self.cutoff).unsqueeze(1)
        accuracy = (predictions == data.y.to(self.device)).float().mean()
        self.batch_metric = torch.cat(
            (self.batch_metric, torch.tensor([[accuracy]], device=self.device)), 
            dim=0
        )

    def compute(self):
        return self.batch_metric.mean()