"""
    gnn_model.py - Neutron Capture GNN

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
from torch_geometric.loader import DataLoader
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, LeakyReLU
from torch_geometric.nn import MessagePassing, GATConv, MetaLayer
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import mlflow.pytorch
from normalizations import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# local includes
from neutron_dataset import NeutronDataset

example_config = {
    "node_feats":           4,
    "node_classes":         1,
    "edge_feats":           1,
    "edge_classes":         1,
    "global_feats":         0,
    "node_output_feats":    0,
    "edge_output_feats":    0,
    "global_output_feats":  0,
    "num_mp":               3,
    "aggr":                 "add",
    "leakiness":            0.1,
}

class GammaFragmentModel(torch.nn.Module):
    
    def __init__(self, cfg):
        """
        """
        super(GammaFragmentModel, self).__init__()

        self.model_config   = cfg
        self.node_input     = self.model_config.get('node_feats', 4)
        self.edge_input     = self.model_config.get('edge_feats', 1)
        self.global_input   = self.model_config.get('global_feats', 1)
        self.node_output    = self.model_config.get('node_output_feats', 2)
        self.edge_output    = self.model_config.get('edge_output_feats', self.edge_input)
        self.global_output  = self.model_config.get('global_output_feats', 1)
        self.aggr           = self.model_config.get('aggr', 'add')
        self.leakiness      = self.model_config.get('leakiness', 0.1)

        self.gatConvs = torch.nn.ModuleList()
        self.edge_updates = torch.nn.ModuleList()

        self.bn_node = torch.nn.ModuleList()
        self.bn_edge = BatchNorm1d(self.edge_input)

        self.num_mp = self.model_config.get('num_mp', 3)

        node_input  = self.node_input
        node_output = self.node_output
        edge_input  = self.edge_input
        edge_output = self.edge_output
        for i in range(self.num_mp):
            self.bn_node.append(BatchNorm(node_input))
            self.gatConvs.append(GATConv(node_input, node_output))
            # self.bn_node.append(BatchNorm(node_output))
            # print(node_input, node_output)
            self.edge_updates.append(
                MetaLayer(edge_model=EdgeLayer(node_input, edge_input, edge_output,
                                    leakiness=self.leakiness)#,
                          #node_model=NodeLayer(node_output, node_output, self.edge_input,
                                                #leakiness=self.leakiness)
                          #global_model=GlobalModel(node_output, 1, 32)
                         )
            )
            node_input = node_output
            edge_input = edge_output

        self.node_classes = self.model_config.get('node_classes', 2)
        self.edge_classes = self.model_config.get('edge_classes', 2)

        self.node_predictor = nn.Linear(node_output, self.node_classes)
        self.edge_predictor = nn.Linear(edge_output, self.edge_classes)

    def forward(self, node_features, edge_indices, edge_features, xbatch):
        """
        """
        x = node_features.view(-1, self.node_input)
        e = edge_features.view(-1, self.edge_input)
        for i in range(self.num_mp):
            x = self.bn_node[i](x)
            # add u and batch arguments for not having error in some old version
            _, e, _ = self.edge_updates[i](x, edge_indices, e, u=None, batch=xbatch)
            x = self.gatConvs[i](x, edge_indices)
            # x = self.bn_node(x)
            x = F.leaky_relu(x, negative_slope=self.leakiness)
        # print(edge_indices.shape)
        x_pred = self.node_predictor(x)
        e_pred = self.edge_predictor(e)

        res = {
            'node_pred': [x_pred],
            'edge_pred': [e_pred]
            }

        return res

class EdgeLayer(nn.Module):
    '''
    An EdgeModel for predicting edge features.
    Example: Parent-Child Edge prediction and EM primary assignment prediction.
    INPUTS:
        DEFINITIONS:
            E: number of edges
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output edge features
            B: number of graphs (same as batch size)
        If an entry i->j is an edge, then we have source node feature
        F^i_x, target node feature F^j_x, and edge features F_e.
        - source: [E, F_x] Tensor, where E is the number of edges
        - target: [E, F_x] Tensor, where E is the number of edges
        - edge_attr: [E, F_e] Tensor, indicating input edge features.
        - global_features: [B, F_u] Tensor, where B is the number of graphs
        (equivalent to number of batches).
        - batch: [E] Tensor containing batch indices for each edge from 0 to B-1.
    RETURNS:
        - output: [E, F_o] Tensor with F_o output edge features.
    '''
    def __init__(self, node_in, edge_in, edge_out, leakiness=0.0):
        super(EdgeLayer, self).__init__()
        # TODO: Construct Edge MLP
        self.edge_mlp = nn.Sequential(
            BatchNorm1d(2 * node_in + edge_in),
            nn.Linear(2 * node_in + edge_in, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeLayer(nn.Module):
    '''
    NodeModel for node feature prediction.
    Example: Particle Classification using node-level features.
    INPUTS:
        DEFINITIONS:
            N: number of nodes
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output node features
            B: number of graphs (same as batch size)
        If an entry i->j is an edge, then we have source node feature
        F^i_x, target node feature F^j_x, and edge features F_e.
        - source: [E, F_x] Tensor, where E is the number of edges
        - target: [E, F_x] Tensor, where E is the number of edges
        - edge_attr: [E, F_e] Tensor, indicating input edge features.
        - global_features: [B, F_u] Tensor, where B is the number of graphs
        (equivalent to number of batches).
        - batch: [E] Tensor containing batch indices for each edge from 0 to B-1.
    RETURNS:
        - output: [C, F_o] Tensor with F_o output node feature
    '''
    def __init__(self, node_in, node_out, edge_in, leakiness=0.0):
        super(NodeLayer, self).__init__()

        self.node_mlp_1 = nn.Sequential(
            BatchNorm1d(node_in + edge_in),
            nn.Linear(node_in + edge_in, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

        self.node_mlp_2 = nn.Sequential(
            BatchNorm1d(node_in + node_out),
            nn.Linear(node_in + node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        from torch_scatter import scatter_mean
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(nn.Module):
    '''
    Global Model for global feature prediction.
    Example: event classification (graph classification) over the whole image
    within a batch.
    Do Hierarchical Pooling to reduce features
    INPUTS:
        DEFINITIONS:
            N: number of nodes
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output node features
            B: number of graphs (same as batch size)
        If an entry i->j is an edge, then we have source node feature
        F^i_x, target node feature F^j_x, and edge features F_e.
        - source: [E, F_x] Tensor, where E is the number of edges
        - target: [E, F_x] Tensor, where E is the number of edges
        - edge_attr: [E, F_e] Tensor, indicating input edge features.
        - global_features: [B, F_u] Tensor, where B is the number of graphs
        (equivalent to number of batches).
        - batch: [E] Tensor containing batch indices for each edge from 0 to B-1.
    RETURNS:
        - output: [C, F_o] Tensor with F_o output node feature
    '''
    def __init__(self, node_in, batch_size, global_out, leakiness=0.0):
        super(GlobalModel, self).__init__()

        self.global_mlp = nn.Sequential(
            BatchNorm1d(node_in + batch_size),
            nn.Linear(node_in + batch_size, global_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(global_out),
            nn.Linear(global_out, global_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(global_out),
            nn.Linear(global_out, global_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        from torch_scatter import scatter_mean
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


train_dataset = NeutronDataset("../data/",4,1,eps=.5)

model = GammaFragmentModel(example_config)
model = model.to(device)

weights = torch.tensor([1,10], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

NUM_GRAPHS_PER_BATCH = 3
train_loader = DataLoader(train_dataset,
    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train(epoch):
    all_preds = []
    all_labels = []
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x.float(),
            batch.edge_index,
            batch.edge_attr.float(),
            batch.batch)
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        optimizer.step()

        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_pred, y_true)}")
    print(f"Accuracy: {accuracy_score(y_pred, y_true)}")
    print(f"Precision: {precision_score(y_pred, y_true)}")
    print(f"Recall: {recall_score(y_pred, y_true)}")
    try:
        roc = roc_auc_score(y_pred, y_true)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")

with mlflow.start_run() as run:
    for epoch in range(500):

        model.train()
        loss = train(epoch=epoch)
        loss = loss.detach().cpu().numpy()
        print(f"Epoch {epoch} | Train loss {loss}")
        mlflow.log_metric(ley="Train loss", value=float(loss), step=epoch)

        model.eval()
        if epoch % 5 == 0:
            pass

        scheduler.step()

    print("Done")

mlflow.pytorch.log_model(model, "model")