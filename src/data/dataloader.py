"""
数据加载器
用于加载眼底视网膜图像和分割标签
支持数据增强和低质量图像合成
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance


class FundusDataset(Dataset):
    """
    眼底图像数据集
    """
    def __init__(
        self,
        image_paths: list,
        mask_paths: list = None,
        transform=None,
        image_size: tuple = (512, 512),
        is_train: bool = True,
        synthesize_low_quality: bool = True
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size
        self.is_train = is_train
        self.synthesize_low_quality = synthesize_low_quality
        
        # 基础变换
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def _synthesize_low_quality(self, img: Image.Image) -> tuple:
        """
        合成低质量图像
        返回高质量原图和低质量图像
        """
        high_quality = img.copy()
        
        # 随机选择退化类型
        deg_types = ['blur', 'noise', 'illumination', 'contrast', 'all']
        deg_type = random.choice(deg_types)
        
        low_quality = high_quality.copy()
        
        if deg_type in ['blur', 'all']:
            # 模糊
            blur_radius = random.uniform(1, 4)
            low_quality = low_quality.filter(ImageFilter.GaussianBlur(blur_radius))
        
        if deg_type in ['noise', 'all']:
            # 添加噪声
            np_img = np.array(low_quality)
            noise_level = random.uniform(0.02, 0.1)
            noise = np.random.normal(0, noise_level * 255, np_img.shape)
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            low_quality = Image.fromarray(np_img)
        
        if deg_type in ['illumination', 'all']:
            # 光照不均
            np_img = np.array(low_quality).astype(np.float32)
            h, w = np_img.shape[:2]
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x, y)
            intensity = 0.3 * np.sin(xx * 2) * np.cos(yy * 2) + 1
            np_img = np_img * intensity[..., np.newaxis]
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)
            low_quality = Image.fromarray(np_img)
        
        if deg_type in ['contrast', 'all']:
            # 低对比度
            contrast_factor = random.uniform(0.4, 0.7)
            enhancer = ImageEnhance.Contrast(low_quality)
            low_quality = enhancer.enhance(contrast_factor)
            
            # 亮度调整
            brightness_factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(low_quality)
            low_quality = enhancer.enhance(brightness_factor)
        
        return high_quality, low_quality
    
    def __getitem__(self, idx: int) -> dict:
        # 加载图像
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.image_size, Image.BILINEAR)
        
        # 加载掩膜（如果有）
        mask = None
        if self.mask_paths and idx < len(self.mask_paths):
            mask_path = self.mask_paths[idx]
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = mask.resize(self.image_size, Image.NEAREST)
                mask = np.array(mask)
                # 确保是单通道
                if len(mask.shape) == 3:
                    mask = mask[..., 0]
                mask = torch.from_numpy(mask).long()
        
        # 合成低质量图像（训练时）
        high_quality_img = img
        low_quality_img = img
        if self.is_train and self.synthesize_low_quality:
            high_quality_img, low_quality_img = self._synthesize_low_quality(img)
        
        # 数据增强
        if self.is_train and self.transform:
            # 应用变换
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            
            high_quality_img = self.transform(high_quality_img)
            low_quality_img = self.transform(low_quality_img)
        else:
            # 基础变换
            high_quality_img = self.to_tensor(high_quality_img)
            low_quality_img = self.to_tensor(low_quality_img)
        
        # 归一化
        high_quality_img = self.normalize(high_quality_img)
        low_quality_img = self.normalize(low_quality_img)
        
        item = {
            'high_quality': high_quality_img,
            'low_quality': low_quality_img
        }
        
        if mask is not None:
            item['mask'] = mask
        
        return item


def get_data_transforms(image_size: tuple = (512, 512)):
    """
    获取训练和验证的数据变换
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return train_transform, val_transform


def create_dataloaders(
    train_image_paths: list,
    train_mask_paths: list = None,
    val_image_paths: list = None,
    val_mask_paths: list = None,
    image_size: tuple = (512, 512),
    batch_size: int = 8,
    num_workers: int = 4
):
    """
    创建训练和验证数据加载器
    """
    train_transform, val_transform = get_data_transforms(image_size)
    
    # 训练集
    train_dataset = FundusDataset(
        train_image_paths,
        train_mask_paths,
        transform=train_transform,
        image_size=image_size,
        is_train=True,
        synthesize_low_quality=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 验证集
    val_loader = None
    if val_image_paths:
        val_dataset = FundusDataset(
            val_image_paths,
            val_mask_paths,
            transform=val_transform,
            image_size=image_size,
            is_train=False,
            synthesize_low_quality=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader
