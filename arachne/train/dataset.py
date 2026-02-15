import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(Path(images_dir))
        self.masks = sorted(Path(masks_dir))

        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_path = Path(self.images_dir, self.images[idx])
            mask_path = Path(self.masks_dir, self.masks[idx])
        
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            if self.transform:
                image = self.trasnform(image)
                mask = self.transform(mask)

            mask = torch.where(mask > 0, 1.0, 0.0)

            return image, mask

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])