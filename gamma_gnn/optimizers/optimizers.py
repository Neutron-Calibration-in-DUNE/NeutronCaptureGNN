"""
Optimizers for gamma_gnn.
"""
import torch.optim as optim

from gamma_gnn.utils.logger import Logger

optimizer_config = {
    "optimizer":    "Adam",
    "learning_rate": 0.01,
    "momentum":      0.9,
    "weight_decay":  0.0001,
    "schedulers":   {
        "ExponentialLR": {
            "gamma": 0.9,
        },
    }
}

class Optimizer:
    """
    A standard optimizer for pytorch models.
    """
    def __init__(self,
        model,
        cfg,

    ):
        self.name = model.name + "_optimizer"
        self.cfg = cfg
        self.logger = Logger(self.name, file_mode='w')
        # set learning rate and momentum

        self.logger.info(f"learning rate set to {self.cfg['learning_rate']}")
        self.logger.info(f"momentum value set to {self.cfg['momentum']}")

        # set the optimizer
        if self.cfg['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.cfg['learning_rate'],
                weight_decay=self.cfg['weight_decay']
            )
            self.logger.info(f"using the Adam optimizer")
        elif self.cfg['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.cfg['learning_rate'],
                momentum=self.cfg['momentum'],
                weight_decay=self.cfg['weight_decay']
            )
        self.schedulers = []
        if "schedulers" in self.cfg.keys():
            for scheduler in self.cfg['schedulers']:
                if scheduler == "ExponentialLR":
                    self.schedulers.append(
                        optim.lr_scheduler.ExponentialLR(
                            self.optimizer, 
                            gamma=self.cfg['schedulers'][scheduler]["gamma"]
                        )
                    )
        
    def zero_grad(self):
        return self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
        for ii in range(len(self.schedulers)):
            self.schedulers[ii].step()
        return