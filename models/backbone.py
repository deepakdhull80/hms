import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import ConfigV1


class BaseBackbone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x , y=None):
        raise NotImplementedError
    
    @torch.no_grad()
    def forward_infer(self, x):
        raise NotImplementedError

class EfficientNet(BaseBackbone):
    def __init__(self, config:ConfigV1) -> None:
        super().__init__()
        self.backbone = None
        # self.loss_fn = nn.KLDivLoss(reduction="batchmean")
        # self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def modify_model(self):
        assert isinstance(self.backbone,nn.Module), "Backbone should be a nn.Module and Efficientnet"
        
        layer = self.backbone.features.pop(0)
        first_conv_blk: nn.Conv2d = layer.pop(0)
        layer.insert(
            0, nn.Conv2d(
                ConfigV1.model.conv_in_channels, 
                first_conv_blk.out_channels, 
                kernel_size = first_conv_blk.kernel_size,
                stride = first_conv_blk.stride,
                padding = first_conv_blk.padding,
                bias = first_conv_blk.bias
                )
            )
        self.backbone.features.insert(0, layer)
        backbone_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_dim, len(ConfigV1.class_columns))
        )
        
    
    def forward(self, x:torch.Tensor, y: torch.Tensor=None):
        out = self.backbone(x)
        if y is None:
            return F.softmax(out, dim=-1)
        
        out = F.softmax(out, dim=-1)
        
        if isinstance(self.loss_fn, nn.KLDivLoss):
            y = y / y.sum(dim=-1).unsqueeze(1)
        
        return self.loss_fn(out, y)

    @torch.no_grad()
    def forward_infer(self, x):
        return self.backbone(x)

class EfficientNet_S(EfficientNet):
    def __init__(self, config: ConfigV1) -> None:
        super().__init__(config)
        self.backbone = torchvision.models.efficientnet_v2_s(torchvision.models.EfficientNet_V2_S_Weights) \
            if config.model.load_pretrained_weights else torchvision.models.efficientnet_v2_s()
        self.modify_model()


class EfficientNet_M(EfficientNet):
    def __init__(self, config: ConfigV1) -> None:
        super().__init__(config)
        self.backbone = torchvision.models.efficientnet_v2_m(torchvision.models.EfficientNet_V2_M_Weights) \
            if config.model.load_pretrained_weights else torchvision.models.efficientnet_v2_m()
        self.modify_model()
