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
import pandas as pd
# local includes
from neutron_dataset import NeutronDataset


def calc_dist(p1,p2):
    dist = 0
    for i in range(len(p1)):
        dist += (p1[i] - p2[i])*(p1[i] - p2[i])
    return np.sqrt(dist)

def cluster_distances(
    input_file
):
    data = pd.read_csv(
        input_file,
        header=0,
        names=['x','y','z','num_electrons','energy','gamma','neutron']
    )
    clusters = []
    temp_cluster = []
    temp_gamma = data['gamma'][0]
    for i in range(len(data['gamma'])):
        if data['gamma'][i] == temp_gamma:
            temp_cluster.append([data['x'][i],data['y'][i],data['z'][i]])
        else:
            clusters.append(temp_cluster)
            temp_gamma = data['gamma'][i]
            temp_cluster = [[data['x'][i],data['y'][i],data['z'][i]]]
    # determine max distances in cluster
    max_distances = []
    for i in range(len(clusters)):
        max_dist = 0
        for j in range(len(clusters[i])):
            for k in range(len(clusters[i])):
                if j != k:
                    dist = calc_dist(clusters[i][j],clusters[i][k])
                    if dist > max_dist:
                        max_dist = dist
        max_distances.append(max_dist)
    return max_distances

distances = []
for i in range(50):
    distances.append(cluster_distances(f"../data/raw/train_{i}.csv"))
flat_list = [item for sublist in distances for item in sublist]

fig, axs = plt.subplots()
axs.hist(flat_list)
plt.show()