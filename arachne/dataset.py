import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF
import random

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(list(Path(images_dir).glob("*.*")))
        self.masks = sorted(list(Path(images_dir).glob("*.*")))

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = Path(self.images_dir, self.images[idx])
        mask_path = Path(self.masks_dir, self.masks[idx])
    
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.where(mask > 0, 1.0, 0.0)

        return image, mask

class TrainTransform:
    def __call__(self, img, mask):
        # Resizing
        img = TF.resize(img, (572, 572))
        mask = TF.resize(mask, (388, 388))

        # Random brightness
        brightness = random.uniform(0.75, 1.25)
        img = TF.adjust_brightness(img, brightness)

        # Rotating randomly
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        mask = torch.where(mask > 0, 1.0, 0.0)

        return img, mask

class ValidationTransform:
    def __call__(self, img, mask):
        # Resizing
        img = TF.resize(img, (572, 572))
        mask = TF.resize(mask, (388, 388))

        # Tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        mask = torch.where(mask > 0, 1.0, 0.0)

        return img, mask