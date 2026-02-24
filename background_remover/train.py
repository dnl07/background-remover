import torch
from torch.utils.data import DataLoader, random_split, Subset
from .unet import UNet
from .dataset import SegmentationDataset, TrainTransform, ValidationTransform
from .printer import info, warning, success
from pathlib import Path

def split_flat(train_images_dir: str, 
          train_masks_dir: str, 
          batch_size: int, 
          verbose=False,
          val_split=0.2          
        ):
    # Flat mode: auto-split into train/val
    full_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform=None)
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(total), [train_size, val_size], generator=generator)

    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, TrainTransform())
    val_dataset = SegmentationDataset(train_images_dir, train_masks_dir, ValidationTransform())

    train_subset = Subset(train_dataset, train_indices.indices)
    val_subset = Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,  num_workers=4, pin_memory=True)

    if verbose:
        info(f"Auto-split: {len(train_subset)} training / {len(val_subset)} validation samples")

    return train_loader, val_loader

def train(train_images_dir: str, 
          train_masks_dir: str, 
          val_images_dir: str | None, 
          val_masks_dir: str | None, 
          epochs: int, 
          batch_size: int, 
          learning_rate: float, 
          early_stopping=False, 
          resume_from=None,
          verbose=False,
          val_split=0.2
        ):
    '''Train the UNet model for background removal.'''

    if val_images_dir and val_masks_dir:
        # Pre-split mode: separate train/val directories
        train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, TrainTransform())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        if verbose:
            info(f"Found {len(train_dataset)} training samples")

        val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, ValidationTransform())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        if verbose:
            info(f"Found {len(val_dataset)} validation samples")
    else:
        train_loader, val_loader = split_flat(train_images_dir, train_masks_dir, batch_size, verbose, val_split)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        if (device.type == "cuda"):
            info("Using GPU")
        else:
            warning("Using CPU")

    # Initialize model, loss function, and optimizer
    model = UNet(num_classes=1).to(device)

    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=device))
        if verbose:
            info(f"Resumed training from {resume_from}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # early stopping
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    if verbose:
        info("Training is starting...")

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for train_images, train_masks in train_loader:
            train_images = train_images.to(device)
            train_masks = train_masks.to(device)

            preds = model(train_images)
            loss = criterion(preds, train_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * train_images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                val_preds = model(val_images)
                loss = criterion(val_preds, val_masks)
                val_loss += loss.item() * val_images.size(0)

        val_loss /= len(val_loader.dataset)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                warning("Early stopping triggered")
                save(model, "models/unet_bg_removal.pth")
                return

        if verbose:
            info(f"Epoch [{epoch + 1}/{epochs}] - Training-Loss: {epoch_loss:.4f} - Val-Loss: {val_loss:.4f}")

        # Save checkpoint after each epoch (overwritten each time as a backup)
        checkpoint_path = Path("models/unet_checkpoint.pth")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
    
    # Save the final trained model
    save(model, "models/unet_bg_removal.pth")

    # Remove checkpoint after successful save
    checkpoint_path = Path("models/unet_checkpoint.pth")
    if checkpoint_path.exists():
        checkpoint_path.unlink()

def save(model, path):
    '''Save the trained model to the specified path.'''
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    i = 1
    while model_path.exists():
        model_path = Path(f"models/unet_bg_removal_{i}.pth")
        i += 1

    torch.save(model.state_dict(), model_path)
    success(f"Model {model_path} saved!")