import torch.nn as nn
from torchvision import models
from .base import BaseExtractor

class MobileNetV2Extractor(BaseExtractor):
    def __init__(self, device=None):
        super().__init__(model_name='mobilenet_v2', device=device)

    def _get_model(self):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Loại bỏ lớp classifier cuối cùng
        return nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
