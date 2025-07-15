from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from PIL import Image

# 1) Build the backbone AND immediately swap in a 2-way head
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# 2) Load your fine-tuned 2-way checkpoint
state = torch.load('checkpoints/resnet50_anime_vs_cartoon.pth', map_location='cpu')
model.load_state_dict(state)



# 3) Set eval mode
model.eval()






#Preprocessing pipeline

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225])
])

labels = {0: 'anime', 1: 'cartoon'}

def predict(image: Image.Image) -> dict:

    #Run the inference on a PIL image.
    #Returns label: str, confidence: float

    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim =1)[0]
        idx = torch.argmax(probs).item()
    return {'label': labels[idx], 'confidence': probs[idx].item()}