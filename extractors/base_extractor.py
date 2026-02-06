import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class BaseExtractor:
    def __init__(self, model_name, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.model_name = model_name
        self.model = self._get_model()
        self.model = self.model.to(self.device)
        self.model.eval()

    def _get_model(self):
        """Mỗi subclass sẽ ghi đè phương thức này để trả về model tương ứng."""
        raise NotImplementedError("Subclasses must implement _get_model")

    def extract(self, dataloader):
        features = []
        labels = []
        paths = []
        
        print(f"--- Trích xuất đặc trưng với {self.model_name} trên {self.device} ---")
        with torch.no_grad():
            for imgs, lbls, pths in tqdm(dataloader, desc=f"Extracting {self.model_name}"):
                imgs = imgs.to(self.device)
                output = self.model(imgs)
                # Flatten kết quả về vector 1D (trừ batch size)
                output = output.view(output.size(0), -1)
                
                features.append(output.cpu().numpy())
                labels.extend(lbls.numpy())
                paths.extend(pths)
                
        return np.vstack(features), np.array(labels), np.array(paths)
