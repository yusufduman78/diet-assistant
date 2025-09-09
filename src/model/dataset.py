import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import json
from pathlib import Path

class FoodDataset(Dataset):
    """
        Image path and label can be loaded using dataset_split.csv
    """
    def __init__(self,data_path,transform=None):
        """

        :param data_path(str): Main dataset path
        :param split_type(str): 'train', 'val' and 'test'
        :param transform(callable,optional): Transform for apply to images
        """
        self.data_path = Path(data_path)
        self.transform = transform

        #read dataset_split.csv
        self.df = pd.read_csv(self.data_path / "dataset_split.csv")

        #class_to_idx
        class_names = sorted(self.df['class_name'].unique())
        self.class_to_idx = {name: i for i,name in enumerate(class_names)}
        self.idx_to_class = {i: name for name,i in self.class_to_idx.items()}

        #save to .json files
        with open('models/class_name.json','w') as f:
            json.dump(self.class_to_idx,f,indent=4)

        print(f"{len(self.df)} images loaded from dataset.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        img_path = Path(row['file_path'])

        #load image by PIL
        from PIL import Image
        image = Image.open(img_path).convert('RGB')

        #transform
        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row['class_name']]
        return image,label
