import torch
import csv
import os
import getpass
from datetime import datetime
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)

from gamma_gnn.utils import Logger

default_config = {
    "embedding_size": 128,
    "attention_heads": 4,
    "layers": 4,
    "top_k_ratio": 0.5,
    "top_k_every_n": 2,
    "dense_neurons": 256
}

class GNN(torch.nn.Module):
    def __init__(self, 
        name,
        feature_size,
        edge_dim,
        cfg=default_config
    ):
        self.name = name
        self.feature_size = feature_size
        self.edge_dim = edge_dim
        self.cfg = cfg
        self.logger = Logger(self.name, file_mode='w')
        super(GNN, self).__init__()
        self.n_layers = cfg["layers"]
        self.top_k_every_n = cfg["top_k_every_n"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(
            self.feature_size, 
            self.cfg["embedding_size"], 
            heads=self.cfg["attention_heads"], 
            dropout=0,
            edge_dim=self.edge_dim,
            beta=True
        ) 

        self.transf1 = Linear(self.cfg["embedding_size"]*self.cfg["attention_heads"], self.cfg["embedding_size"])
        self.bn1 = BatchNorm1d(self.cfg["embedding_size"])

        # Other layers
        for i in range(self.cfg["layers"]):
            self.conv_layers.append(
                TransformerConv(
                    self.cfg["embedding_size"], 
                    self.cfg["embedding_size"], 
                    heads=self.cfg["attention_heads"], 
                    dropout=0,
                    edge_dim=self.edge_dim,
                    beta=True
                )
            )
            self.transf_layers.append(Linear(self.cfg["embedding_size"]*self.cfg["attention_heads"], self.cfg["embedding_size"]))
            self.bn_layers.append(BatchNorm1d(self.cfg["embedding_size"]))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(self.cfg["embedding_size"], ratio=self.cfg["top_k_ratio"]))
            
        # Linear layers
        self.linear1 = Linear(self.cfg["embedding_size"]*2, self.cfg["dense_neurons"])
        self.linear2 = Linear(self.cfg["dense_neurons"], int(self.cfg["dense_neurons"]/2))  
        self.linear3 = Linear(int(self.cfg["dense_neurons"]/2), 1)  
        self.output = torch.nn.Sigmoid()
        # device for the model
        self.device = None

    def set_device(self,
        device
    ):
        self.device = device
        self.to(device)

    def forward(self, data):
        data = data.to(self.device)
        x = data.x.float() 
        edge_attr = data.edge_attr.float()
        edge_index = data.edge_index
        batch_index = data.batch
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                )
                # Add current representation
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
    
        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        #x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        #x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)
        x = self.output(x)

        return x
    def save_model(self,
        flag:   str=''
    ):
        # save meta information
        if not os.path.isdir(f"models/{self.name}/"):
            os.makedirs(f"models/{self.name}/")
        output = f"models/{self.name}/" + self.name
        if flag != '':
            output += "_" + flag
        if not os.path.exists("models/"):
            os.makedirs("models/")
        meta_info = [[f'Meta information for model {self.name}']]
        meta_info.append(['date:',datetime.now().strftime("%m/%d/%Y %H:%M:%S")])
        meta_info.append(['user:', getpass.getuser()])
        meta_info.append(['user_id:',os.getuid()])
        system_info = self.logger.get_system_info()
        if len(system_info) > 0:
            meta_info.append(['System information:'])
            for item in system_info:
                meta_info.append([item,system_info[item]])
            meta_info.append([])
        meta_info.append(['Model configuration:'])
        meta_info.append([])
        for item in self.cfg:
            meta_info.append([item, self.cfg[item]])
        meta_info.append([])
        meta_info.append(['Model dictionary:'])
        for item in self.state_dict():
            meta_info.append([item, self.state_dict()[item].size()])
        meta_info.append([])
        with open(output + "_meta.csv", "w") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows(meta_info)
        # save config
        cfg = [[item, self.cfg[item]] for item in self.cfg]
        with open(output+".cfg", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(cfg)
        # save parameters
        torch.save(
            {
            'model_state_dict': self.state_dict(), 
            'model_config': self.cfg
            }, 
            output + "_params.ckpt"
        )
    
    def load_model(self,
        model_file:   str=''
    ):
        self.logger.info(f"Attempting to load model checkpoint from file {model_file}.")
        try:
            checkpoint = torch.load(model_file)
            self.cfg = checkpoint['model_config']
            self.construct_model()
            # register hooks
            self.register_forward_hooks()
            self.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            self.logger.error(f"Unable to load model file {model_file}: {e}.")
            raise ValueError(f"Unable to load model file {model_file}: {e}.")
        self.logger.info(f"Successfully loaded model checkpoint.")