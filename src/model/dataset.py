import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json
from pathlib import Path
from PIL import Image


class FoodDataset(Dataset):
    """
        Image path and label can be loaded using dataset_split.csv
    """
    def __init__(self, data_path, transform=None, split=None):
        """
        :param data_path (str or Path): Main dataset path
        :param transform (callable, optional): Transform to apply to images
        :param split (str, optional): 'train', 'val', 'test' -> filter dataset by split
        """
        self.data_path = Path(data_path)
        self.transform = transform

        # read dataset_split.csv
        self.df = pd.read_csv(self.data_path / "dataset_split.csv")

        if split:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # class_to_idx
        class_names = sorted(self.df['class_name'].unique())
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        # save to .json file
        metadata_dir = self.data_path.parent / "01_metadata"
        metadata_dir.mkdir(exist_ok=True, parents=True)
        with open(metadata_dir / 'class_names.json', 'w', encoding="utf-8") as f:
            json.dump(self.class_to_idx, f, indent=4, ensure_ascii=False)

        print(f"{len(self.df)} images loaded from dataset ({split if split else 'all'}).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        path_parts = Path(row['file_path']).parts
        correct_path_from_root = Path(*path_parts[1:])
        project_root = Path(__file__).resolve().parents[2]
        img_path = project_root / 'data' / correct_path_from_root

        # load image by PIL
        image = Image.open(img_path).convert('RGB')

        # apply transform
        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row['class_name']]
        return image, label


def create_data_loaders(data_path, train_transform, val_transform,
                        batch_size=64, num_workers=4, class_weights=True):
    """
        Create data loaders for train, val, test
    :param data_path: dataset directory containing dataset_split.csv
    :param train_transform: torchvision transforms for train set
    :param val_transform: torchvision transforms for val & test
    :param batch_size: dataloader batch size
    :param num_workers: number of workers for dataloader
    :param class_weights: return class weights (for imbalanced datasets)
    :return: train_loader, val_loader, test_loader, (weights if class_weights=True)
    """

    train_dataset = FoodDataset(data_path, transform=train_transform, split="train")
    val_dataset = FoodDataset(data_path, transform=val_transform, split="val")
    test_dataset = FoodDataset(data_path, transform=val_transform, split="test")


    weights = None
    if class_weights:
        class_counts = train_dataset.df['class_name'].value_counts().sort_index()
        num_samples = len(train_dataset)
        num_classes = len(class_counts)
        class_weights_list = [num_samples / (num_classes * count) for count in class_counts]
        weights = torch.tensor(class_weights_list, dtype=torch.float)
        weights = weights / weights.sum()  #normalizing

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    if class_weights:
        return train_loader, val_loader, test_loader, weights
    else:
        return train_loader, val_loader, test_loader


# test block
if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    data_path = "../../data/processed/01_final_dataset"
    train_loader, val_loader, test_loader, weights = create_data_loaders(
        data_path=data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=32
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Class weights shape: {weights.shape}")


