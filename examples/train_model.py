"""
Train a simple GNN model on 
gamma data
"""
import numpy as np
from matplotlib import pyplot as plt

from gamma_gnn.dataset.gamma import GammaDataset
from gamma_gnn.utils.loader import Loader
from gamma_gnn.models import GNN
from gamma_gnn.optimizers import Optimizer
from gamma_gnn.losses import LossHandler
from gamma_gnn.metrics import MetricHandler
from gamma_gnn.utils.callbacks import CallbackHandler
from gamma_gnn.trainer import Trainer


if __name__ == "__main__":
    
    # now generate graph data
    gamma_dataset = GammaDataset(
        "gamma",
        "data/raw/gammas.npz",
        "data/"
    )
    # create the data loader
    gamma_loader = Loader(
        gamma_dataset, 
        batch_size=16,
        test_split=0.1,
        test_seed=0,
        validation_split=0.1,
        validation_seed=0,
        num_workers=8
    )

    """
    Construct the gamma Model, specify the loss and the 
    optimizer and metrics.
    """
    gamma_config = {
        "embedding_size": 32,
        "attention_heads": 2,
        "layers": 4,
        "top_k_ratio": 0.8,
        "top_k_every_n": 3,
        "dense_neurons": 64
    }
    gamma_model = GNN(
        name = 'gamma_test',
        feature_size=gamma_dataset[0].x.shape[1],
        edge_dim=gamma_dataset[0].edge_attr.shape[1],
        cfg  = gamma_config
    ) 

    # create loss, optimizer and metrics
    gamma_optimizer_config = {
        "optimizer":    "SGD",
        "learning_rate": 0.01,
        "momentum":      0.8,
        "weight_decay":  0.0001,
        "schedulers":   {
            "ExponentialLR": {
                "gamma": 1.0,
            },
        }
    }
    gamma_optimizer = Optimizer(
        model=gamma_model,
        cfg=gamma_optimizer_config
    )

    # create criterions
    gamma_loss_config = {
        'BCELoss':   {
            'alpha':    1.0,
            'reduction':'mean',
        },
    }
    gamma_loss = LossHandler(
        name="gamma_loss",
        cfg=gamma_loss_config,
    )
    
    # create metrics
    gamma_metric_config = {
        'BinaryAccuracy': {
            'cutoff':   0.5,
        },
        'Precision': {
            'cutoff':   0.5,
        },
        'Recall': {
            'cutoff':   0.5,
        },
        'F1Score': {
            'cutoff':   0.5,
        },
        'ROCAUC': {
            'cutoff':   0.5,
        },
        # 'OutputSaver':  {},
        # 'TargetSaver':  {},
    }
    gamma_metrics = MetricHandler(
        "gamma_metric",
        cfg=gamma_metric_config,
    )

    # create callbacks
    callback_config = {
        'loss':   {'criterion_list': gamma_loss},
        'metric': {'metrics_list':   gamma_metrics},
    }
    gamma_callbacks = CallbackHandler(
        "gamma_callbacks",
        callback_config
    )

    # create trainer
    gamma_trainer = Trainer(
        model=gamma_model,
        criterion=gamma_loss,
        optimizer=gamma_optimizer,
        metrics=gamma_metrics,
        callbacks=gamma_callbacks,
        metric_type='test',
        gpu=True,
        gpu_device=0
    )
    
    gamma_trainer.train(
        gamma_loader,
        epochs=25,
        checkpoint=25
    )