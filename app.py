import os
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

# ---- Load Classes ----
DATA_PATH = "dataset"
if os.path.exists(DATA_PATH):
    dataset = datasets.ImageFolder(DATA_PATH)
    class_names = dataset.classes
else:
    class_names = []
    print(f"Warning: Dataset path '{DATA_PATH}' not found.")

# ---- Load Model ----
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
# Handle case where class_names is empty to avoid error during initialization if dataset is missing
num_classes = len(class_names) if class_names else 2 # Default to 2 if empty to avoid crash, though model won't work right
model.fc = nn.Linear(num_features, num_classes)

MODEL_PATH = "avengers_resnet50.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
else:
    print(f"Warning: Model file '{MODEL_PATH}' not found. Please train the model first.")

model.eval()

# ---- Transform ----
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- Routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            try:
                # Read file into memory
                img_bytes = file.read()
                
                # Create PIL Image from bytes
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
                # Transform
                img_t = transform(img).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    outputs = model(img_t)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                prediction = class_names[pred.item()] if class_names else "Unknown"
                confidence = round(conf.item() * 100, 2)
                
                # Encode for display (Base64)
                mime_type = file.mimetype or "image/jpeg"
                encoded_img = base64.b64encode(img_bytes).decode('utf-8')
                image_path = f"data:{mime_type};base64,{encoded_img}"
                
            except Exception as e:
                print(f"Error processing image: {e}")

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)