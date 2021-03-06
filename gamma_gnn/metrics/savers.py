"""
Generic saver metric class for gamma_gnn.
"""
import torch
import torch.nn as nn

from gamma_gnn.metrics import GenericMetric

class OutputSaver(GenericMetric):
    
    def __init__(self,
        name:   str='output_saver',
        output_shape:   tuple=(),
        target_shape:    tuple=(),
    ):
        """
        output Saver
        """
        super(OutputSaver, self).__init__(
            name,
            output_shape,
            target_shape
        )
         # create empty tensors for epoch 
        self.batch_output = torch.empty(
            size=(0,*self.output_shape), 
            dtype=torch.float, device=self.device
        )

        self.epoch_output = None
        
    def reset_batch(self):
        if len(self.batch_output) != 0:
            self.epoch_output = self.batch_output
        self.batch_output = torch.empty(
            size=(0,*self.output_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_output = torch.cat((self.batch_output, outputs[0]),dim=0)

    def compute(self):
        pass

class TargetSaver(GenericMetric):
    
    def __init__(self,
        name:   str='target_saver',
        output_shape:   tuple=(),
        target_shape:    tuple=(),
    ):
        """
        Input Saver
        """
        super(TargetSaver, self).__init__(
            name,
            output_shape,
            target_shape
        )
         # create empty tensors for epoch 
        self.batch_input = torch.empty(
            size=(0,*self.target_shape), 
            dtype=torch.float, device=self.device
        )
        self.epoch_input = None
        
    def reset_batch(self):
        if len(self.batch_input) != 0:
            self.epoch_input = self.batch_input
        self.batch_input = torch.empty(
            size=(0,*self.target_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_input = torch.cat((self.batch_input, data[0].to(self.device)),dim=0)

    def compute(self):
        pass
        