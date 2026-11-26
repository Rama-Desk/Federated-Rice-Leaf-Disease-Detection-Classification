"""MobileNetV3-Small model definition used by both centralized checkpoints."""
from torch import nn
from torchvision.models import mobilenet_v3_small


def build_model(num_classes: int = 10):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model
