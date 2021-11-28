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
from torch_geometric.data import Data, Dataset
import torch_geometric
import csv
import os
from sklearn.cluster import DBSCAN

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class NeutronDataset(Dataset):
    """
    """
    
    def __init__(self, 
        root,
        num_node_features:  int,
        num_edge_features:  int,
        eps:    float=3,
        transform=None,
        pre_transform=None,
    ):
        """
            root = where the dataset should be stored, this folder
            is split into raw_dir and processed_dir
        """
        self.dbscan = DBSCAN(eps=eps)
        super(NeutronDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
            If these files are found in the raw directory, then
            downloading is not triggered.
        """
        return [f'train_{i}.csv' for i in range(500)]
        #return [f'train_{i}.csv' for i in range(250)]
        #return [f'train_{i}.csv' for i in range(250,500)]

    @property
    def processed_file_names(self):
        """
            If these files are found in raw_dir, processing is skipped
        """
        self.data = pd.read_csv(
            self.raw_paths[0],
            header=0,
            names=['x','y','z','num_electrons','energy','gamma','neutron']).reset_index()
        return [f'train_{i}.pt' for i in range(500)]
        #return ['']

    def download(self):
        pass

    def process(self):
        for i in range(500):
            self.data = pd.read_csv(
                self.raw_paths[i],
                header=0,
                names=['x','y','z','num_electrons','energy','gamma','neutron'])
            # construct node features
            node_feats = self._get_node_features()
            # get the adjacency info
            edge_index = self._get_adjacency_info()
            # get the edge features
            edge_feats = self._get_edge_features()
            # get the labels
            labels = self._get_labels()

            data_obj = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=labels)

            torch.save(data_obj, 
                os.path.join(self.processed_dir, f'train_{i}.pt'))
            #torch.save(data_obj, 
            #    os.path.join(self.processed_dir, f'test_{i}.pt'))

    def _get_node_features(self):
        """
            Construct the node features 
        """
        self.x = []
        self.node_feats = []
        for i in range(len(self.data)):
            self.x.append([
                self.data['x'][i],    # x position
                self.data['y'][i],    # y position
                self.data['z'][i]     # z position
            ])
            self.node_feats.append([
                self.data['x'][i],    # x position
                self.data['y'][i],    # y position
                self.data['z'][i],    # z position
                self.data['num_electrons'][i]     # number of electrons
            ])
        return torch.tensor(np.asarray(self.node_feats), dtype=torch.float)
    
    def _get_adjacency_info(self):
        """
            First, run dbscan on the positions
        """
        self.dbscan.fit(self.x)
        # get the labels
        labels = self.dbscan.labels_
        # construct list of cluster indices
        temp_label = labels[0]
        self.edge_labels = []
        temp_labels = []
        self.edge_index = [[],[]]
        for i in range(len(labels)):
            if labels[i] == temp_label:
                temp_labels.append(i)
            else:
                for j in range(len(temp_labels)):
                    for k in range(len(temp_labels)):
                        if j != k:
                            self.edge_index[0].append(temp_labels[j])
                            self.edge_index[1].append(temp_labels[k])
                self.edge_labels.append(temp_labels)
                temp_label = labels[i]
                temp_labels = [i]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def _get_edge_features(self):
        """
            Adjacency info must be constructed first
        """
        self.edge_feats = []
        for i in range(len(self.edge_index[0])):
            self.edge_feats.append(
                [self._distance(
                    self.x[self.edge_index[0][i]],
                    self.x[self.edge_index[1][i]]
                )]
            )
        return torch.tensor(np.asarray(self.edge_feats), dtype=torch.float)

    def _get_labels(self):
        """
            Construct training labels
        """
        self.labels = []
        for i in range(len(self.data)):
            self.labels.append(self.data['gamma'][i])
        # for all edges constructed, determine whether they
        # should exist or not
        self.edge_preds = []
        for i in range(len(self.edge_index[0])):
            if self.data['gamma'][self.edge_index[0][i]] == self.data['gamma'][self.edge_index[1][i]]:
                self.edge_preds.append([1])
            else:
                self.edge_preds.append([0])
        return torch.tensor(np.asarray(self.edge_preds), dtype=torch.float)
        #return torch.tensor(np.asarray(self.labels), dtype=torch.int64)

    def _distance(self,
        p1,
        p2
    ):
        """
            Compute the Euclidean distance
        """
        distance = 0
        for i in range(len(p1)):
            distance += (p1[i] - p2[i])*(p1[i] - p2[i])
        return np.sqrt(distance)

    def len(self):
        return 500

    def get(self,idx):
        data = torch.load(os.path.join(self.processed_dir, f'train_{idx}.pt'))
        return data
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

dataset = NeutronDataset("../data",4,1,eps=50)