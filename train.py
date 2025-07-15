import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Dataset + split
    data_dir = os.path.join("archive", "Training Data")
    full_ds = ImageFolder(data_dir, transform=train_tfms)
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_ds.dataset.transform = train_tfms  # ensure train aug
    val_ds.dataset.transform   = val_tfms    # lighter val transforms

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(full_ds.classes))
    model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train/val loop (as before)…
    num_epochs = 5
    for epoch in range(num_epochs):
        # training block …
        # validation block …
        pass

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/resnet50_anime_vs_cartoon.pth")
    print("Checkpoint saved.")

if __name__ == "__main__":
    main()
