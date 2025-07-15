# from torchvision import models, transforms
# import torch
# from PIL import Image

#Loading pretrained model and replace final layer
# model = models.resnet50(pretrained = True)
# model.fc = torch.nn.Linear(model.fc.in_features, 2)
# checkpoint = torch.load('resnet50-0676ba61.pth', map_location='cpu')
# model.load_state_dict(checkpoint)
# model.eval()

from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights

import torch
import torch.nn as nn
from PIL import Image
# 1) Build the *full* ResNet (with 1000-way head)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 2) Load checkpoint into a dict
state = torch.load('resnet50-0676ba61.pth', map_location='cpu')

# 3) Remove the classifier weights from that dict
state.pop('fc.weight', None)
state.pop('fc.bias',   None)

# 4) Load the rest (strict=False lets you skip missing keys)
model.load_state_dict(state, strict=False)

# 5) Now swap in your 2-way head
model.fc = nn.Linear(model.fc.in_features, 2)

model.eval()






#Preprocessing pipeline

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225])
])

labels = {0: 'cartoon', 1: 'anime'}

def predict(image: Image.Image) -> dict:

    #Run the inference on a PIL image.
    #Returns label: str, confidence: float

    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim =1)[0]
        idx = torch.argmax(probs).item()
    return {'label': labels[idx], 'confidence': probs[idx].item()}