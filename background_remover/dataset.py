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

        # 2. Random crop
        if random.random() < 0.4:
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(450, 450))
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
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

        # Random affine
        angle = random.uniform(-20, 20)
        translate = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-5, 5)

        img = TF.affine(img, angle, translate, scale, shear, interpolation=Image.BILINEAR)
        mask = TF.affine(mask, angle, translate, scale, shear, interpolation=Image.NEAREST)

        # Perspective warp
        if random.random() < 0.2:
            startpoints, endpoints = T.RandomPerspective.get_params(512, 512, distortion_scale=0.4)
            img = TF.perspective(img, startpoints, endpoints)
            mask = TF.perspective(mask, startpoints, endpoints)

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