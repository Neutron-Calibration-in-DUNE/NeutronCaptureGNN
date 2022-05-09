"""
Class for converting neutron root files into npz
file for the GNN
"""
import numpy as np
import uproot
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset, random_split

from gamma_gnn.utils.logger import Logger

class NeutronDataset:
    """
    """
    def __init__(self,
        data_dir,
        data_files: list=[],
    ):
        self.logger = Logger('neutron_dataset', output='both', file_mode='w')
        self.data_dir = data_dir
        self.data_files = data_files
        self.sp_x, self.sp_y, self.sp_z = [], [], []
        self.summed_adc, self.peak_adc, self.mean_adc, self.sigma_adc = [], [], [], []
        self.gamma_id, self.neutron_id, self.gamma_energy = [], [], []

        self.compile_events()

    def compile_events(self):
        self.logger.info(f"Compiling events from {len(self.data_files)} files.")
        sp_x, sp_y, sp_z = [], [], []
        summed_adc, peak_adc, mean_adc, sigma_adc = [], [], [], []
        gamma_id, neutron_id, gamma_energy = [], [], []
        for file in self.data_files:
            data = uproot.open(self.data_dir+file)['ana/reco_neutrons'].arrays(library="np")
            sp_x.append(data['sp_x'])
            sp_y.append(data['sp_y'])
            sp_z.append(data['sp_z'])
            summed_adc.append(data['summed_adc'])
            peak_adc.append(data['peak_adc'])
            mean_adc.append(data['mean_adc'])
            sigma_adc.append(data['sigma_adc'])
            gamma_id.append(data['gamma_id'])
            neutron_id.append(data['neutron_id'])
            gamma_energy.append(data['gamma_energy'])
        sp_x = np.concatenate(sp_x)
        sp_y = np.concatenate(sp_y)
        sp_z = np.concatenate(sp_z)
        summed_adc = np.concatenate(summed_adc)
        peak_adc = np.concatenate(peak_adc)
        mean_adc = np.concatenate(mean_adc)
        sigma_adc = np.concatenate(sigma_adc)
        gamma_id = np.concatenate(gamma_id)
        neutron_id = np.concatenate(neutron_id)
        gamma_energy = np.concatenate(gamma_energy)
        # go through and split events into individual gammas
        for ii in range(len(sp_x)):
            unique_gammas = np.unique(gamma_id[ii])
            for jj, g in enumerate(unique_gammas):
                mask = (gamma_id[ii] == g)
                self.sp_x.append(sp_x[ii][mask])
                self.sp_y.append(sp_y[ii][mask])
                self.sp_z.append(sp_z[ii][mask])
                self.summed_adc.append(summed_adc[ii][mask])
                self.peak_adc.append(peak_adc[ii][mask])
                self.mean_adc.append(mean_adc[ii][mask])
                self.sigma_adc.append(sigma_adc[ii][mask])
                self.gamma_id.append(gamma_id[ii][mask])
                self.neutron_id.append(neutron_id[ii][mask])
                self.gamma_energy.append(gamma_energy[ii][mask])
        total_gammas = np.round(np.concatenate(self.gamma_energy),6)
        unique_gammas = np.unique(total_gammas)
        self.unique_gammas = {
            unique_gammas[ii]: [
                ii, len(total_gammas[(total_gammas == unique_gammas[ii])])
            ] 
            for ii in range(len(unique_gammas))
        }

        self.num_events = len(self.gamma_id)
        self.logger.info(f"Successfully compiled {self.num_events} events from {len(self.data_files)} files.")

    def generate_training_set(self,
        output_dir: str,
        output_file: str,
    ):
        # find min and max
        meta_dict = {
            "num_events": self.num_events,
            "sp_x_norm": [np.min(np.concatenate(self.sp_x)),np.max(np.concatenate(self.sp_x))],
            "sp_y_norm": [np.min(np.concatenate(self.sp_y)),np.max(np.concatenate(self.sp_y))],
            "sp_z_norm": [np.min(np.concatenate(self.sp_z)),np.max(np.concatenate(self.sp_z))],
            "summed_adc_norm": [np.min(np.concatenate(self.summed_adc)),np.max(np.concatenate(self.summed_adc))],
            "peak_adc_norm": [np.min(np.concatenate(self.peak_adc)),np.max(np.concatenate(self.peak_adc))],
            "mean_adc_norm": [np.min(np.concatenate(self.mean_adc)),np.max(np.concatenate(self.mean_adc))],
            "sigma_adc_norm": [np.min(np.concatenate(self.sigma_adc)),np.max(np.concatenate(self.sigma_adc))],
        }
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        np.savez(
            output_dir+output_file,
            meta=meta_dict,
            gammas=self.unique_gammas,
            sp_x=self.sp_x,
            sp_y=self.sp_y,
            sp_z=self.sp_z,
            summed_adc=self.summed_adc,
            peak_adc=self.peak_adc,
            mean_adc=self.mean_adc,
            sigma_adc=self.sigma_adc,
            gamma_id=self.gamma_id,
            neutron_id=self.neutron_id,
            gamma_energy=self.gamma_energy,
        )