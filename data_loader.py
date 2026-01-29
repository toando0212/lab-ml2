import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, num_classes=10):
        self.df = pd.read_csv(csv_file)
        # Lọc lấy 10 lớp đầu tiên (ClassId 0-9)
        self.df = self.df[self.df['ClassId'] < num_classes].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['ClassId']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, self.df.iloc[idx]['Path']

def get_dataloader(csv_file, root_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = GTSRBDataset(csv_file, root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
