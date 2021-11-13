import uproot
import numpy as np
import csv
import os

def make_dir(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

def generate_training_files(input_root_file,
    folder_name='../data/train/',
):
    make_dir(folder_name)
    f = uproot.open(input_root_file)
    vals = f['ana/neutron'].arrays()
    for i in range(len(vals['event_id'])):
        # node features
        edep_x = vals['edep_x'][i]
        edep_y = vals['edep_y'][i]
        edep_z = vals['edep_z'][i]
        edep_num_e = vals['edep_num_electrons'][i]
        # labels
        edep_energy = vals['edep_energy'][i]
        edep_gamma_ids = vals['edep_gamma_ids'][i]
        edep_neutron_ids = vals['edep_neutron_ids'][i]

        data = []
        for j in range(len(edep_x)):
            if edep_num_e[j] != 0:
                data.append(
                    [
                        edep_x[j],
                        edep_y[j],
                        edep_z[j],
                        edep_num_e[j],
                        edep_energy[j],
                        edep_gamma_ids[j],
                        edep_neutron_ids[j]
                    ]
                )

        with open(folder_name+"train_{}.csv".format(i),"w") as file:
            writer = csv.writer(file,delimiter=",")
            writer.writerows(data)

if __name__ == "__main__":
    generate_training_files("~/physics/neutrino_physics/NeutronDataset_500_100.root")

