import torch.nn as nn
from torchvision import models
from .base import BaseExtractor

class ResNet50Extractor(BaseExtractor):
    def __init__(self, device=None):
        super().__init__(model_name='resnet50', device=device)

    def _get_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Loại bỏ lớp FC cuối cùng
        return nn.Sequential(*list(model.children())[:-1])
