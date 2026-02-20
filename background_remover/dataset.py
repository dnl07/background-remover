import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

# Dataset class for loading images and masks for training and validation
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(list(Path(images_dir).glob("*.*")))
        self.masks = sorted(list(Path(masks_dir).glob("*.*")))

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

# Data augmentation for training
class TrainTransform:
    def __call__(self, img, mask):
        # Resizing
        img = TF.resize(img, (512, 512), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (512, 512), interpolation=Image.NEAREST)

        # Random brightness
        brightness = random.uniform(0.75, 1.25)
        img = TF.adjust_brightness(img, brightness)

        # Random contrast
        contrast = random.uniform(0.75, 1.25)
        img = TF.adjust_contrast(img, contrast)

        # Random saturation
        saturation = random.uniform(0.75, 1.25)
        img = TF.adjust_saturation(img, saturation)

        # Gaussian blur
        if random.random() < 0.3:
            img = TF.gaussian_blur(img, kernel_size=3)

        # Rotating randomly
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        # Random horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return img, mask

# Data transformation for validation (no augmentation)
class ValidationTransform:
    def __call__(self, img, mask):
        # Resizing
        img = TF.resize(img, (512, 512))
        mask = TF.resize(mask, (512, 512))

        # Tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        mask = torch.where(mask > 0, 1.0, 0.0)

        return img, mask