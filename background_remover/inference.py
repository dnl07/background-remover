from datetime import datetime
import torch
from typing import Optional 
from PIL import Image
import torchvision.transforms.functional as TF
from .unet import UNet
import numpy as np
from .printer import success
from pathlib import Path

def inference(image_path, model_path, with_mask):
    '''Run inference on a single image using a trained UNet model.'''

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size

    img_resized = TF.resize(img, (572, 572))
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    # Load the model
    model = UNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()
        mask = mask.squeeze(0).cpu()
    
    # Post-process the mask and save the result
    mask_pil = TF.to_pil_image(mask)
    mask_resized = mask_pil.resize(orig_size, resample=Image.NEAREST)

    mask_np = np.array(mask_resized)
    mask_np = (mask_np > 0.5).astype(np.uint8)

    alpha = mask_np * 255

    img_np = np.array(img)

    rgba = np.dstack((img_np, alpha))
    result = Image.fromarray(rgba)

    return result, mask_resized

def save_images(output_dir, image, mask: Optional[Image.Image]):
    '''Save the inferred image and mask to the specified output directory.'''

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save the image and mask with unique filenames
    image_path = f"{output_dir}/foreground_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(image_path)
    if mask:
        mask_path = f"{output_dir}/mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        mask.save(mask_path)
    success("Image saved")