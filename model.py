from torchvision import models, transforms
import torch
from PIL import Image

#Loading pretrained model and replace final layer
model = models.resnet50(pretrained = True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('resnet50_anime_vs_cartoon.pth', map_location='cpu'))
model.eval()

#Preprocessing pipeline

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225])
])

labels = {0: 'anime', 1: 'cartoon'}

def predict(image: Image.image) -> dict:

    #Run the inference on a PIL image.
    #Returns label: str, confidence: float

    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim =1)[0]
        idx = torch.argmax(probs).item()
    return {'label': labels[idx], 'confidence': probs[idx].item()}