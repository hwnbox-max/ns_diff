"""
Dataset Loaders for Shapes3D and CelebA-HQ
按照论文4.1.1节的数据集描述实现
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import h5py
import os
from typing import Tuple, Dict, List, Optional
import pandas as pd


class Shapes3DDataset(Dataset):
    """
    Shapes3D Dataset
    包含480,000张图像,6个ground-truth因子:
    - floor_hue: 地板色调
    - wall_hue: 墙壁色调
    - object_hue: 物体色调
    - scale: 尺度
    - shape: 形状
    - orientation: 方向

    数据来源: https://github.com/deepmind/3d-shapes
    """

    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 transform: Optional[transforms.Compose] = None,
                 target_attribute: str = 'shape'):
        """
        Args:
            data_path: 3dshapes.h5文件路径
            split: 'train' 或 'test'
            train_ratio: 训练集比例
            transform: 图像变换
            target_attribute: 目标分类属性
        """
        self.data_path = data_path
        self.split = split
        self.target_attribute = target_attribute

        # 加载数据
        with h5py.File(data_path, 'r') as f:
            self.images = f['images'][:]  # [480000, 64, 64, 3]
            self.labels = f['labels'][:]  # [480000, 6]

        # 因子名称和维度
        self.factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        self.factor_sizes = [10, 10, 10, 8, 4, 15]

        # 划分训练/测试集
        total_samples = len(self.images)
        train_size = int(total_samples * train_ratio)

        if split == 'train':
            self.images = self.images[:train_size]
            self.labels = self.labels[:train_size]
        else:
            self.images = self.images[train_size:]
            self.labels = self.labels[train_size:]

        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # 获取目标属性的索引
        self.target_idx = self.factor_names.index(target_attribute)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Returns:
            image: 变换后的图像 [3, H, W]
            target: 目标类别标签
            concepts: 所有因子作为概念 [6]
        """
        image = self.images[idx]  # [64, 64, 3]
        factors = self.labels[idx]  # [6]

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 目标标签
        target = int(factors[self.target_idx])

        # 归一化概念到[0, 1]
        concepts = torch.tensor([
            factors[i] / self.factor_sizes[i] for i in range(6)
        ], dtype=torch.float32)

        return image, target, concepts


class CelebAHQDataset(Dataset):
    """
    CelebA-HQ Dataset
    高分辨率(256x256)人脸数据集,30,000张图像

    使用8个主要属性作为概念:
    - Bangs, Beard, Smiling, Male, Young, Eyeglasses, Wavy_Hair, Wearing_Hat

    数据来源: https://github.com/tkarras/progressive_growing_of_gans
    """

    def __init__(self,
                 image_dir: str,
                 attr_file: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 transform: Optional[transforms.Compose] = None,
                 target_attribute: str = 'Male',
                 concept_attributes: Optional[List[str]] = None):
        """
        Args:
            image_dir: 图像文件夹路径
            attr_file: 属性CSV/TXT文件路径
            split: 'train' 或 'test'
            train_ratio: 训练集比例
            transform: 图像变换
            target_attribute: 目标分类属性
            concept_attributes: 用作概念的属性列表
        """
        self.image_dir = image_dir
        self.split = split
        self.target_attribute = target_attribute

        # 默认概念属性 (论文中提到的8个)
        if concept_attributes is None:
            self.concept_attributes = [
                'Bangs', 'Beard', 'Smiling', 'Male',
                'Young', 'Eyeglasses', 'Wavy_Hair', 'Wearing_Hat'
            ]
        else:
            self.concept_attributes = concept_attributes

        # 加载属性文件
        self.attr_df = self._load_attributes(attr_file)

        # 获取图像文件列表
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png'))
        ])

        # 过滤掉没有属性标注的图像
        self.image_files = [
            f for f in self.image_files
            if f.replace('.png', '').replace('.jpg', '') in self.attr_df.index
        ]

        # 划分训练/测试集
        train_size = int(len(self.image_files) * train_ratio)
        if split == 'train':
            self.image_files = self.image_files[:train_size]
        else:
            self.image_files = self.image_files[train_size:]

        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def _load_attributes(self, attr_file: str) -> pd.DataFrame:
        """
        加载属性文件
        支持CelebA原始格式或CSV格式
        """
        if attr_file.endswith('.csv'):
            df = pd.read_csv(attr_file, index_col=0)
        else:
            # CelebA原始txt格式
            with open(attr_file, 'r') as f:
                lines = f.readlines()

            # 第一行是数量,第二行是属性名
            num_samples = int(lines[0].strip())
            attr_names = lines[1].strip().split()

            # 解析数据
            data = []
            for line in lines[2:]:
                parts = line.strip().split()
                img_name = parts[0]
                attrs = [int(x) for x in parts[1:]]
                data.append([img_name] + attrs)

            df = pd.DataFrame(data, columns=['image_id'] + attr_names)
            df.set_index('image_id', inplace=True)

            # CelebA使用-1/1编码,转换为0/1
            df = (df + 1) // 2

        return df

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Returns:
            image: 变换后的图像 [3, H, W]
            target: 目标类别标签 (0或1)
            concepts: 概念属性向量 [num_concepts]
        """
        img_file = self.image_files[idx]
        img_id = img_file.replace('.png', '').replace('.jpg', '')

        # 加载图像
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 获取属性
        attrs = self.attr_df.loc[img_id]

        # 目标标签
        target = int(attrs[self.target_attribute])

        # 概念向量
        concepts = torch.tensor([
            float(attrs[attr]) for attr in self.concept_attributes
        ], dtype=torch.float32)

        return image, target, concepts


def get_dataloader(dataset_name: str,
                   data_path: str,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    获取训练和测试数据加载器

    Args:
        dataset_name: 'shapes3d' 或 'celeba-hq'
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    if dataset_name == 'shapes3d':
        train_dataset = Shapes3DDataset(
            data_path=data_path,
            split='train',
            **kwargs
        )
        test_dataset = Shapes3DDataset(
            data_path=data_path,
            split='test',
            **kwargs
        )
    elif dataset_name == 'celeba-hq':
        image_dir = kwargs.get('image_dir', data_path)
        attr_file = kwargs.get('attr_file')

        train_dataset = CelebAHQDataset(
            image_dir=image_dir,
            attr_file=attr_file,
            split='train',
            **kwargs
        )
        test_dataset = CelebAHQDataset(
            image_dir=image_dir,
            attr_file=attr_file,
            split='test',
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


# 示例用法
if __name__ == "__main__":
    # Shapes3D示例
    print("Testing Shapes3D Dataset...")
    shapes3d_loader, _ = get_dataloader(
        dataset_name='shapes3d',
        data_path='/path/to/3dshapes.h5',
        batch_size=16
    )

    for images, targets, concepts in shapes3d_loader:
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Concepts shape: {concepts.shape}")
        break

    # CelebA-HQ示例
    print("\nTesting CelebA-HQ Dataset...")
    celeba_loader, _ = get_dataloader(
        dataset_name='celeba-hq',
        data_path='/path/to/celeba-hq',
        image_dir='/path/to/celeba-hq/images',
        attr_file='/path/to/celeba-hq/attributes.txt',
        batch_size=16
    )

    for images, targets, concepts in celeba_loader:
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Concepts shape: {concepts.shape}")
        break