import torch
from PIL import Image
import torchvision.transforms.functional as TF
from .unet import UNet
import numpy as np

def inference(image_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = Image.open(image_path).convert("RGB")
    orig_size = img.size

    img_resized = TF.resize(img, (572, 572))
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    model = UNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()
        mask = mask.squeeze(0).cpu()
    
    mask_pil = TF.to_pil_image(mask)
    mask_resized = mask_pil.resize(orig_size, resample=Image.NEAREST)

    mask_np = np.array(mask_resized)
    mask_np = (mask_np > 0.5).astype(np.uint8)

    alpha = mask_np * 255

    img_np = np.array(img)

    rgba = np.dstack((img_np, alpha))
    result = Image.fromarray(rgba)
    result.save("foreground.png")

    print("Image saved")