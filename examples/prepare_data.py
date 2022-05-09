"""
Prepare graph data from original root files
"""
from telnetlib import GA
import numpy as np
import torch

from gamma_gnn.dataset.neutron import NeutronDataset
from gamma_gnn.dataset.gamma import GammaDataset

if __name__ == "__main__":

    # first construct npz file
    data_dir = "../../neutron/neutron_data/"
    data_files = [
        f"neutrons{ii}.root"
        for ii in range(10)
    ]
    neutron_dataset = NeutronDataset(
        data_dir, data_files
    )
    neutron_dataset.generate_training_set(
        "data/raw/",
        "gammas.npz"
    )

    # now generate graph data
    gamma_dataset = GammaDataset(
        "data/raw/gammas.npz",
        "data/",
        edge_dist=0.01
    )


