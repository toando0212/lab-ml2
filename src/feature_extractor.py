import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor:
    def __init__(self, model_name='resnet50', device='cpu'):
        self.device = device
        self.model_name = model_name
        
        if model_name == 'resnet50':
            # Load ResNet50 pretrained
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Loại bỏ lớp fully connected cuối cùng để lấy đặc trưng
            self.model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'inception_v3':
            # Load InceptionV3 pretrained
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
            model.fc = nn.Identity() # Identity cho fc layer
            self.model = model
            self.model.aux_logits = False # Tắt aux_logits cho inference
        else:
            raise ValueError("Model không hỗ trợ. Chỉ hỗ trợ 'resnet50' hoặc 'inception_v3'.")
            
        self.model = self.model.to(device)
        self.model.eval()

    def extract(self, dataloader):
        features = []
        labels = []
        paths = []
        
        with torch.no_grad():
            from tqdm import tqdm
            for imgs, lbls, pths in tqdm(dataloader, desc=f"Extracting {self.model_name}"):
                imgs = imgs.to(self.device)
                
                if self.model_name == 'resnet50':
                    output = self.model(imgs)
                    output = output.view(output.size(0), -1) # Flatten (batch, 2048)
                else: # inception_v3
                    output = self.model(imgs)
                
                features.append(output.cpu().numpy())
                labels.extend(lbls.numpy())
                paths.extend(pths)
                
        import numpy as np
        return np.vstack(features), np.array(labels), np.array(paths)
