"""
Train a simple GNN model on 
gamma data
"""
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import shutil

from gamma_gnn.dataset.gamma import GammaDataset
from gamma_gnn.utils.loader import Loader
from gamma_gnn.models import GNN
from gamma_gnn.optimizers import Optimizer
from gamma_gnn.losses import LossHandler
from gamma_gnn.metrics import MetricHandler
from gamma_gnn.utils.callbacks import CallbackHandler
from gamma_gnn.trainer import Trainer

from mango import scheduler, Tuner

def run_one_training(params):
    params = params[0]
    # now generate graph data
    gamma_dataset = GammaDataset(
        "gamma",
        "data/raw/gammas.npz",
        "data/"
    )
    # create the data loader
    gamma_loader = Loader(
        gamma_dataset, 
        batch_size=params["batch_size"],
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
        "embedding_size":   params["embedding_size"],
        "attention_heads":  params["attention_heads"],
        "layers":           params["layers"],
        "top_k_ratio":      params["top_k_ratio"],
        "top_k_every_n":    params["top_k_every_n"],
        "dense_neurons":    params["dense_neurons"],
    }
    model_name = f'gamma_{params["embedding_size"]}_{params["attention_heads"]}_{params["layers"]}_{params["top_k_ratio"]}_{params["top_k_every_n"]}_{params["dense_neurons"]}'
    model_name += f'_{params["learning_rate"]}_{params["momentum"]}_{params["weight_decay"]}_{params["scheduler_gamma"]}'
    now = datetime.now()
    os.makedirs(model_name + f"_{now}/")

    gamma_model = GNN(
        name = model_name,
        feature_size=gamma_dataset[0].x.shape[1],
        edge_dim=gamma_dataset[0].edge_attr.shape[1],
        cfg  = gamma_config
    ) 

    # create loss, optimizer and metrics
    gamma_optimizer_config = {
        "optimizer":    "SGD",
        "learning_rate": params["learning_rate"],
        "momentum":      params["momentum"],
        "weight_decay":  params["weight_decay"],
        "schedulers":   {
            "ExponentialLR": {
                "gamma": params["scheduler_gamma"],
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
        epochs=100,
        checkpoint=25
    )
    shutil.move("plots/", model_name + f"_{now}/")
    shutil.move("models/", model_name + f"_{now}/")
    shutil.copy("data/raw/gammas.npz", model_name + f"_{now}/")
    print(f"Best loss for this training: {gamma_trainer.best_loss}")
    return [gamma_trainer.best_loss]

if __name__ == "__main__":

    HYPERPARAMETERS = {
        "batch_size": [32, 64, 128],
        "learning_rate": [0.1, 0.05, 0.01, 0.001],
        "weight_decay": [0.0001, 0.00001, 0.001],
        "momentum": [0.9, 0.8, 0.5],
        "scheduler_gamma": [0.995, 0.9, 0.8, 0.5, 1],
        "embedding_size": [8, 16, 32, 64, 128],
        "attention_heads": [1, 2, 3, 4],
        "layers": [3],
        "dropout_rate": [0.2, 0.5, 0.9],
        "top_k_ratio": [0.2, 0.5, 0.8, 0.9],
        "top_k_every_n": [1, 2, 3],
        "dense_neurons": [16, 32, 64, 128, 256]
    }

    print("Running hyperparameter search...")
    config = dict()
    config["optimizer"] = "Bayesian"
    config["num_iteration"] = 100

    tuner = Tuner(HYPERPARAMETERS, 
                objective=run_one_training,
                conf_dict=config) 
    results = tuner.minimize()