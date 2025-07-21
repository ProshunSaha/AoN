# evaluate.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1) Build & load your fine-tuned model ---
def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    # Create ResNet-50 backbone + 2-way head
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load trained weights
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# --- 2) Define test-time preprocessing (must match training) ---
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])

if __name__ == "__main__":
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3) Prepare test DataLoader ---
    # ImageFolder will automatically:
    #   • find subfolders under "data/test" (e.g. anime/, cartoon/) as class labels
    #   • recursively load all images within those subfolders
    test_dataset = datasets.ImageFolder("data/test", transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- 4) Load the model ---
    model = load_model("checkpoints/resnet50_anime_vs_cartoon.pth", device)

    # --- 5) Run inference over test set ---
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # --- 6) Compute & print metrics ---
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc:.4f}\n")

    print("=== Classification Report ===")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.classes,
        digits=4
    ))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
