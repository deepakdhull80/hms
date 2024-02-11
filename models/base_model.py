import torch
import torch.nn as nn

from common.utils import my_import
from config.config import ConfigV1
from models.backbone import BaseBackbone

class Classifier(nn.Module):
    def __init__(self, config: ConfigV1):
        super().__init__()
        self.config = config
        self.model: BaseBackbone = my_import(config.model.model_clazz)(config)
    
    def forward(self, x, y = None):
        if y is not None:
            return self.model.forward(x, y)
        
        return self.model.forward_infer(x)
    
    def get_optim(self) -> torch.optim.Optimizer:
        return my_import(self.config.trainer_config.optim_clazz)(self.model.parameters(), lr=self.config.trainer_config.lr)