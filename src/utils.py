"""
    utils.py - Neutron Capture GNN

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
from torch_geometric.nn import MessagePassing, GATConv, MetaLayer
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
# local includes
from neutron_dataset import NeutronDataset