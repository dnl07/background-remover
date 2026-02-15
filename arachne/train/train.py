import torch
from torch.utils.data import DataLoader
from ..unet import UNet
from .dataset import SegmentationDataset, transform

def train(images_dir, masks_dir, epochs, batch_size, learning_rate):
    train_dataset = SegmentationDataset(images_dir, masks_dir, transform)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(num_classes=1).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() + images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), "unet_bg_removal.pth")
    print("Model saved!")