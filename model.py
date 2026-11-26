"""Model definition matching the supplied FedRice-XAI centralized checkpoint."""
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def CentralizedRiceNet(num_classes: int = 10):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model
