import torch
from torch.utils.data import DataLoader
from .unet import UNet
from .dataset import SegmentationDataset, TrainTransform, ValidationTransform
from .printer import info, warning, success
from pathlib import Path

def train(train_images_dir, 
          train_masks_dir, 
          val_images_dir, 
          val_masks_dir, 
          epochs, 
          batch_size, 
          learning_rate, 
          early_stopping=False, 
          resume_from=None,
          verbose=False
        ):
    '''Train the UNet model for background removal.'''

    # Load datasets and create data loaders
    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, TrainTransform())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if verbose:
        info(f"Found {len(train_dataset)} training samples")

    val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, ValidationTransform())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        info(f"Found {len(val_dataset)} validation samples")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        if (device == "cuda"):
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
    
    # Save the trained model
    save(model, "models/unet_bg_removal.pth")

def save(model, path):
    '''Save the trained model to the specified path.'''
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    i = 1
    while model_path.exists():
        model_path = Path(f"models/unet_bg_removal_{i}.pth")
        i += 1

    torch.save(model.state_dict(), model_path)
    success("Model saved!")