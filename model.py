"""Model definitions matching the supplied centralized and personalized checkpoints."""
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def CentralizedRiceNet(num_classes: int = 10):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

class ImprovedPersonalizedFedRice(nn.Module):
    """MobileNetV3-Small shared feature extractor with five private client heads."""
    def __init__(self, num_clients: int = 5, num_classes: int = 10):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.personal_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(576, 1024), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(1024, num_classes))
            for _ in range(num_clients)
        ])
    def forward(self, x, client_id: int = 0):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.personal_heads[client_id](x)
