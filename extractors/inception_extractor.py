import torch.nn as nn
from torchvision import models
from .base_extractor import BaseExtractor

class InceptionV3Extractor(BaseExtractor):
    def __init__(self, device=None):
        super().__init__(model_name='inception_v3', device=device)

    def _get_model(self):
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()
        model.aux_logits = False
        return model
