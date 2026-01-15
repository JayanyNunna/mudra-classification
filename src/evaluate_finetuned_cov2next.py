import torch
from timm import create_model
from torchvision import transforms
import numpy as np
from PIL import Image
import joblib
import torch.nn.functional as F

# 1. Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# 3. Create model architecture first (must match your training config)
model = create_model("convnext_tiny", pretrained=False, num_classes=len(class_names))

# 4. Load fine-tuned weights
model.load_state_dict(torch.load("kuchipudi_mudra_model.pth", map_location=device))
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

svm = joblib.load("svm_mudra_model.pkl")

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(input_tensor)  # [1, 768, 7, 7]
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))  # → [1, 768, 4, 4]
        flattened = pooled.view(pooled.size(0), -1).cpu().numpy()  # → [1, 768*4*4] = 12288
with torch.no_grad():
    features = model.forward_features(input_tensor)  # [1, 768, 7, 7]
    pooled = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))  # → [1, 768, 4, 4]
    flattened = pooled.view(pooled.size(0), -1).cpu().numpy()  # → [1, 768*4*4] = 12288



# Test it
predict("mushti_webcam.jpg")