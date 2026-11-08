from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_iris(bgr_or_gray: np.ndarray, size: int = 224) -> np.ndarray:
    if bgr_or_gray.ndim == 3:
        img = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        img = bgr_or_gray
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = _clahe.apply(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.stack([img] * 3, axis=-1)


def build_train_transform(size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Rotate(limit=15, border_mode=cv2.BORDER_REPLICATE, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.95, 1.05),
                rotate=0,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform(size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class IrisDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform: Optional[A.Compose] = None, size: int = 224):
        self.samples = samples
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read {path}")
        img = preprocess_iris(img, self.size)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            img = (img - mean) / std
        return img, label
