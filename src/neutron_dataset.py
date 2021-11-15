"""
    neutron_dataset.py - Neutron Capture GNN dataset creation

    A colleciton of python classes for implementing graph neural networks in
    neutron capture studies for the DUNE experiment.

    Nicholas Carrara [nmcarrara@ucdavis.edu], Yashwanth Bezawada [ybezawada@ucdavis.edu]
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
# pytorch includes
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import csv

class NeutronDataset(InMemoryDataset):
    """
    """
    
    def __init__(self, 
        num_node_features:  int,
        num_edge_features:  int,
        transform=None,
    ):
        """
        """
        super(NeutronDataset, self).__init__('.', transform, None, None)
        #---------------------------------------------------------------
        # list of nodes and their feature values
        self.x = torch.tensor([], dtype=torch.float)    # (x,y,z)
        # list denoting the edges, the format is
        # [[n1,n2,n3,n4,n5,...],
        #  [m1,m2,m3,m4,m5,...]]
        # where (n1,m1) are two nodes connected by an edge.
        self.edge_index = torch.tensor([[],[]], dtype=torch.long)
        #   list of edges and their features
        self.edge_atr   = torch.tensor([], dtype=torch.float)
        # construct the data object
        self.data = Data(x=self.x, edge_index=self.edge_index)
        #---------------------------------------------------------------


    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)