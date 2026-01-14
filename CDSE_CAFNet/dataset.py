import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

# ----------------  transforms  ----------------
class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image),
                'label': F.to_tensor(label)}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        label = F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        return {'image': image, 'label': label}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            image = F.hflip(image)
            label = F.hflip(label)
        return {'image': image, 'label': label}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            image = F.vflip(image)
            label = F.vflip(label)
        return {'image': image, 'label': label}

class RandomColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < 0.5:
            image = self.transform(image)
        return {'image': image, 'label': label}

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean, self.std = mean, std

    def __call__(self, data):
        image, label = data['image'], data['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

# ----------------  Dataset  ----------------
class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size=448, mode='train'):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        # 保证 32 倍数
        self.size = (self.size[0] - self.size[0] % 32 if self.size[0] % 32 else self.size[0],
                     self.size[1] - self.size[1] % 32 if self.size[1] % 32 else self.size[1])

        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.gts    = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)
                              if f.lower().endswith('.png')])
        assert len(self.images) == len(self.gts), "图像和标签数量不匹配"
        self.mode = mode
        self._init_transforms()

    def _init_transforms(self):
        if self.mode == 'train':
            self.transform = transforms.Compose([
                Resize(self.size),
                RandomColorJitter(),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize(self.size),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        try:
            image = self.rgb_loader(self.images[idx])
            label = self.binary_loader(self.gts[idx])
            data = {'image': image, 'label': label}
            return self.transform(data)
        except Exception as e:
            print(f"加载样本 {idx} 失败: {str(e)}，使用备用样本")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

# ----------------  TestDataset  ----------------
class TestDataset(Dataset):
    def __init__(self, image_root, gt_root, size=448):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.size = (self.size[0] - self.size[0] % 32 if self.size[0] % 32 else self.size[0],
                     self.size[1] - self.size[1] % 32 if self.size[1] % 32 else self.size[1])

        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.gts    = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)
                              if f.lower().endswith('.png')])
        assert len(self.images) == len(self.gts), "测试集图像和标签数量不匹配"

        self.transform = transforms.Compose([
            Resize(self.size),
            ToTensor(),
            Normalize()
        ])
        self.index=0

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = self.transform({'image': image, 'label': label})
        filename = os.path.basename(self.images[idx])
        return {'image': data['image'],
                'label': data['label'],
                'filename': filename}

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def load_data(self):
        """顺序读取下一张图，供单线程测试"""
        if not hasattr(self, 'index'):
            self.index = 0
        if self.index >= len(self):
            raise StopIteration

        # 读取
        image = self.rgb_loader(self.images[self.index])
        label = self.binary_loader(self.gts[self.index])
        name  = os.path.basename(self.images[self.index])
        self.index += 1

        # 统一 transform（Resize+ToTensor+Normalize）
        data  = self.transform({'image': image, 'label': label})
        image = data['image'].unsqueeze(0)          # 加 batch 维
        gt_np = np.array(label, dtype=np.float32)   # 原始尺寸留给后面 interpolate
        return image, gt_np, name