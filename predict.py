import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import os
import torch.nn.functional as F

DATA_PATH = "dataset"
TEST_FOLDER = "bulktest"

# Load class names
dataset = datasets.ImageFolder(DATA_PATH)
class_names = dataset.classes

# Load ResNet50 (same as training)
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model.load_state_dict(torch.load("avengers_resnet50.pth", map_location="cpu"))
model.eval()

# MUST match training normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if not os.path.exists(TEST_FOLDER):
    print("Create a folder named 'bulk_test' and add images.")
    exit()

print("\n===== BULK PREDICTIONS =====\n")

for filename in os.listdir(TEST_FOLDER):
    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".avif")):
        image_path = os.path.join(TEST_FOLDER, filename)

        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        print(f"{filename} → {class_names[predicted.item()]} "
              f"({confidence.item()*100:.2f}%)")