import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, image_root, classes, mode='train', image_size=224):
        """
        :param csv_path: CSV文件路径（train.csv / val.csv / test.csv）
        :param image_root: 原始图像根目录
        :param classes: 多标签类别名列表，如 ['RC', 'RP', 'other']
        :param mode: train / val / test
        :param image_size: 图片尺寸
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.classes = classes
        self.mode = mode
        self.image_size = image_size

        self.transforms = self._build_transforms()

    def _build_transforms(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=(0.2, 1.8), contrast=0.3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_name"])

        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        # 多标签：转为 tensor，如 [1, 0, 1]
        labels = torch.tensor([row[cls] for cls in self.classes], dtype=torch.float32)

        return image, labels
