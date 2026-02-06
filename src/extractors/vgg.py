import torch.nn as nn
from torchvision import models
from .base import BaseExtractor

class VGG16Extractor(BaseExtractor):
    def __init__(self, device=None):
        super().__init__(model_name='vgg16', device=device)

    def _get_model(self):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Lấy đặc trưng sau lớp classifier đầu tiên (4096 chiều)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:1])
        return model
