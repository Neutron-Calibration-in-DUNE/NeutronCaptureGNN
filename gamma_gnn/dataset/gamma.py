"""
Test for torch geometric with neutrons
"""
import numpy as np
from matplotlib import pyplot as plt
import collections
import uproot
import os
import torch
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import pandas as pd
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset


class GammaDataset(Dataset):
    """
    """
    def __init__(self,
        name,
        input_file,
        graph_dir,
        transform=None,
        pre_transform=None,
        edge_dist:  float=-1,
    ):
        self.name = name
        self.input_file = input_file
        self.data = np.load(self.input_file, allow_pickle=True)
        self.graph_dir = graph_dir
        if not os.path.isdir(self.graph_dir + "raw/"):
            os.makedirs(self.graph_dir + "raw/")
        if not os.path.isdir(self.graph_dir + "processed/"):
            os.makedirs(self.graph_dir + "processed/")

        self.meta = self.data['meta'].item()
        self.gammas = self.data['gammas'].item()

        self.sp_x = self.data['sp_x']
        self.sp_y = self.data['sp_y']
        self.sp_z = self.data['sp_z']
        self.summed_adc = self.data['summed_adc']
        self.peak_adc = self.data['peak_adc']
        self.mean_adc = self.data['mean_adc']
        self.sigma_adc = self.data['sigma_adc']
        self.gamma_id = self.data['gamma_id']
        self.neutron_id = self.data['neutron_id']
        self.gamma_energy = self.data['gamma_energy']

        self.sp_x_min = self.meta['sp_x_norm'][0]
        self.sp_x_max = self.meta['sp_x_norm'][1]
        self.sp_y_min = self.meta['sp_y_norm'][0]
        self.sp_y_max = self.meta['sp_y_norm'][1]
        self.sp_z_min = self.meta['sp_z_norm'][0]
        self.sp_z_max = self.meta['sp_z_norm'][1]

        self.summed_adc_min = self.meta['summed_adc_norm'][0]
        self.summed_adc_max = self.meta['summed_adc_norm'][1]
        self.peak_adc_min = self.meta['peak_adc_norm'][0]
        self.peak_adc_max = self.meta['peak_adc_norm'][1]
        self.mean_adc_min = self.meta['mean_adc_norm'][0]
        self.mean_adc_max = self.meta['mean_adc_norm'][1]
        self.sigma_adc_min = self.meta['sigma_adc_norm'][0]
        self.sigma_adc_max = self.meta['sigma_adc_norm'][1]

        self.edge_dist = edge_dist
        # determine edge feaures
        self.__get_edge_features = self._get_edge_lengths

        super(GammaDataset, self).__init__(graph_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.input_file

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = np.load(self.input_file, allow_pickle=True)

        train_files = len([
            f for f in listdir(self.graph_dir + "/processed/") 
            if isfile(join(self.graph_dir + "/processed/"))
        ])
        return [f'data_{i}.pt' for i in range(train_files)]

    def download(self):
        pass

    def process(self):
        """
        Process the data in the raw directories and
        generate the processed .pt files.
        """
        for ii in range(len(self.data['sp_x'])):

            # normalize the data
            x = (self.sp_x[ii] - self.sp_x_min)/(self.sp_x_max - self.sp_x_min)
            y = (self.sp_y[ii] - self.sp_y_min)/(self.sp_y_max - self.sp_y_min)
            z = (self.sp_z[ii] - self.sp_z_min)/(self.sp_z_max - self.sp_z_min)
            summed = (self.summed_adc[ii] - self.summed_adc_min)/(self.summed_adc_max - self.summed_adc_min)
            mean =   (self.mean_adc[ii]   - self.mean_adc_min)/(self.mean_adc_max - self.mean_adc_min)
            peak =   (self.peak_adc[ii]   - self.peak_adc_min)/(self.mean_adc_max - self.peak_adc_min)
            sigma =  (self.sigma_adc[ii]  - self.sigma_adc_min)/(self.sigma_adc_max - self.sigma_adc_min)
            energy = self.gamma_energy[ii]

            # construct the features, indices and labels
            node_feats = torch.tensor(np.vstack((x,y,z,summed,mean,peak,sigma)).T, dtype=torch.float)
            # determine edge distance
            if self.edge_dist == -1:
                edge_index = self._get_edge_index_full(len(x))
            else:
                edge_index = self._get_edge_index_dist(x, y, z)
            edge_feats = self.__get_edge_features(x,y,z,edge_index)
            label      = self.__get_graph_labels(energy)

            # create the gnn data object
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=label,
            )
            # save the data object
            torch.save(
                data,
                os.path.join(self.processed_dir,f'data_{ii}.pt')
            )
    
    def _get_edge_index_full(self, 
        num_nodes
    ):
        """
        returns torch tensor of edge labels
        for a fully connected node list
        """
        e1 = collections.deque([ii for ii in range(num_nodes)])
        e2 = collections.deque([ii for ii in range(num_nodes)])
        edge1 = []
        edge2 = []
        if num_nodes == 1:
            return torch.tensor(
                [[0],[0]],
                dtype=torch.long
            )
        for ii in range(num_nodes-1):
            edge1 += list(e1)
            e2.rotate(1)
            edge2 += list(e2)
        return torch.tensor(
            [edge1,edge2],
            dtype=torch.long
        )

    def _get_edge_index_dist(self,
        x, y, z
    ):
        """
        Returns a list of edges based on wether each 
        pair of points are a distance < self.edge_dist.
        """
        edge1 = []
        edge2 = []
        for ii in range(len(x)):
            for jj in range(len(x)):
                if self._get_length(
                    x[ii],y[ii],z[ii],
                    x[jj],y[jj],z[jj]
                ) < self.edge_dist:
                    edge1.append(ii)
                    edge2.append(jj)
        return torch.tensor(
            [edge1,edge2],
            dtype=torch.long
        )
    def _get_length(self,
        x1, y1, z1,
        x2, y2, z2
    ):
        return np.sqrt(
            (x1-x2)*(x1-x2) + 
            (y1-y2)*(y1-y2) + 
            (z1-z2)*(z1-z2)
        )

    def _get_edge_lengths(self,
        x, y, z,
        edge_index
    ):
        """
        Get the lengths between each pair of points
        in the list of x, y, z values.
        """
        lengths = []
        if len(edge_index) == 1:
            return torch.tensor([[0],[0]], dtype=torch.float)
        for ii in range(len(edge_index[0])):
            lengths.append(
                [self._get_length(
                    x[edge_index[0][ii]],y[edge_index[0][ii]],z[edge_index[0][ii]],
                    x[edge_index[1][ii]],y[edge_index[1][ii]],z[edge_index[1][ii]]
                ),
                self._get_length(
                    x[edge_index[0][ii]],y[edge_index[0][ii]],z[edge_index[0][ii]],
                    x[edge_index[1][ii]],y[edge_index[1][ii]],z[edge_index[1][ii]]
                )]
            )
        return torch.tensor(lengths, dtype=torch.float)

    def __get_graph_labels(self, 
        energy
    ):
        labels = []
        if np.round(energy[0],6) == 0.004745:
            labels.append(1)
        else:
            labels.append(0)
        return torch.tensor(labels, dtype=torch.long)
    
    def len(self):
        return self.meta['num_events']

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))   