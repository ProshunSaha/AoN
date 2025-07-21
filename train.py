import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.models import resnet50, ResNet50_Weights

def main():
    # 1) Device and hyperparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size       = 32
    num_epochs       = 5
    lr               = 1e-4
    oversample_ratio = 1.5  # ← anime samples will be 1.5× as “likely” as cartoons

    # 2) Transforms
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    # 3) Dataset & split
    data_dir = os.path.join("archive", "Training Data")
    full_ds  = ImageFolder(data_dir, transform=None)
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # assign per‐split transforms
    train_ds.dataset.transform = train_tfms
    val_ds.dataset.transform   = val_tfms

    # 4) Build the sampler for oversampling “anime” (label 0)
    all_labels   = torch.tensor([label for _, label in train_ds])
    num_anime    = (all_labels == 0).sum().item()
    num_cartoon  = (all_labels == 1).sum().item()

    # per-class weights
    weight_anime   = oversample_ratio / num_anime
    weight_cartoon = 1.0           / num_cartoon

    sample_weights = torch.zeros(len(all_labels), dtype=torch.double)
    sample_weights[all_labels == 0] = weight_anime
    sample_weights[all_labels == 1] = weight_cartoon

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 5) DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,        # ← use sampler instead of shuffle
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # 6) Model, loss, optimizer
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(full_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 7) Training & validation loop
    for epoch in range(1, num_epochs+1):
        # — Training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # — Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch}/{num_epochs} — Loss: {epoch_loss:.4f}  Val Acc: {val_acc:.4f}")

    # 8) Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/resnet50_anime_vs_cartoon.pth")
    print("✅ Finished training and saved checkpoint.")

if __name__ == "__main__":
    main()
