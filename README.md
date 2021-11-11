# Neutron Capture GNN

<p align="center">
  <img height="150" src="https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_static/img/pyg1.svg?sanitize=true" />
</p>

This repository hosts a GNN implementation using pytorch_geometric for studying neutron capture processes in LArTPCs, specifically in the DUNE experiment.

## Installation

We assume that you have a CUDA capable GPU in your local machine.  
### CUDA toolkit
To install the latest CUDA toolkit, you can either run the installation script:
```
source scripts/install_cuda.sh
```
or follow these steps:
```
wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda_11.5.0_495.29.05_linux.run
sudo sh cuda_11.5.0_495.29.05_linux.run
```
### Anaconda and Python 3.8
Once you have the CUDA toolkit, you'll need to install pytorch and pytorch-geometric.  First however you'll need an appropriate conda evironment with python=3.8.  There is an additional script which will take care of this for you, the steps are the following:
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
chmod +x Anaconda3-2021.05-Linux-x86_64.sh
./Anaconda3-2021.05-Linux-x86_64.sh
```
After the installation, you'll want to create a new python environment:
```
conda create -n pygnn python=3.8
```
### Pytorch and Pytorch Geometric
Then, you can use conda to install pytorch:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Finally, you can install pytorch_geometric with the following:
```
conda install pyg -c pyg -c conda-forge
```
