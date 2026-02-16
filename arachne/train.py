import torch
from torch.utils.data import DataLoader
from .unet import UNet
from .dataset import SegmentationDataset, TrainTransform, ValidationTransform

def train(train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, epochs, batch_size, learning_rate):
    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, TrainTransform())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, ValidationTransform())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(num_classes=1).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for train_images, train_masks in train_loader:
            train_images = train_images.to(device)
            train_masks = train_masks.to(device).unsqueeze(1)

            preds = model(train_images)
            loss = criterion(preds, train_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() + train_images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device).unsqueeze(1)

                val_preds = model(val_images)
                loss = criterion(val_preds, val_masks)
                val_loss += loss.item() * val_images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{epochs}] - Traing-Loss: {epoch_loss:.4f} - Val-Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), "unet_bg_removal.pth")
    print("Model saved!")