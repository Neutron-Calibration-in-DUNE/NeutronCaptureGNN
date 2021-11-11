"""
    gnn_model.py - Neutron Capture GNN

    A colleciton of python classes for implementing graph neural networks in
    neutron capture studies for the DUNE experiment.

    Nicholas Carrara [nmcarrara@ucdavis.edu], Yashwanth Bezawada [ybezawada@ucdavis.edu]
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# pytorch includes
import torch
from torch_geometric.data import Data
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
# local includes
from neutron_dataset import NeutronDataset


class GammaFragmentModel(torch.nn.Module):
    """
    """

    def __init__(self):
        """
        """
        super(GammaFragmentModel, self).__init__()

    def forward(self):
        """
        """
        pass

class GammaFragmentGNN:
    """
    """

    def __init__(self,
        dataset:       NeutronDataset,
        optimizer:  str='Adam',
        learning_rate:  float=1e-1,
    ):
        """
        """
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset    = dataset
        self.data       = self.dataset[0]
        self.gnn        = GammaFragmentModel().to(self.device)
        self.optimizer_name = 'Adam'
        self.learning_rate  = learning_rate
        # set optimizer
        self.optimizer = getattr(
            torch.optim,
            self.optimizer_name
        )(self.gnn.parameters(), lr=self.learning_rate)

        self.epochs = 200

    def train_step(self,
    ) -> None:
        """
        """
        # run the training
        self.gnn.train()
        # resest the gradients
        self.optimizer.zero_grad()
        # calculate the loss
        F.nll_loss(
            self.gnn()[self.data.train_mask], 
            self.data.y[self.data.train_mask]
        ).backward()
        #
        self.optimizer.step()

    @torch.no_grad()
    def train(self,
        epochs: int=1
    ) -> None:
        """
        """
        self.epochs = epochs
        for epoch in range(self.epochs):
            self.train_step()

    def test(self,
    ) -> None:
        """
        """
        self.gnn.eval()
        logits = self.gnn()
        mask1 = self.data['train_mask']
        pred1 = logits[mask1].max(1)[1]
        acc1 = pred1.eq(self.data.y[mask1]).sum().item() / mask1.sum().item()
        mask = self.data['test_mask']
        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc1,acc

    