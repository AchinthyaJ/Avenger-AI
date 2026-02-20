import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import random
import numpy as np

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 🔥 SET YOUR DATASET FOLDER HERE
DATA_PATH = "dataset"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

# ---- Data Augmentation ----
# Training: Heavy augmentation to prevent overfitting
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation: No augmentation, just resize and center crop
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use ImageFolder twice to apply different transforms
train_dataset_full = datasets.ImageFolder(DATA_PATH, transform=train_transform)
val_dataset_full = datasets.ImageFolder(DATA_PATH, transform=val_transform)

# Split indices
num_train = len(train_dataset_full)
indices = list(range(num_train))
split = int(0.8 * num_train)
random.shuffle(indices)

train_idx, val_idx = indices[:split], indices[split:]

train_dataset = Subset(train_dataset_full, train_idx)
val_dataset = Subset(val_dataset_full, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = train_dataset_full.classes
print(f"Classes: {class_names}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ---- Load Pretrained ResNet50 ----
model = models.resnet50(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 (the last convolutional block) and fc for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

criterion = nn.CrossEntropyLoss()
# Use a lower learning rate since we are fine-tuning deeper layers
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': LEARNING_RATE / 10},
    {'params': model.fc.parameters(), 'lr': LEARNING_RATE}
], lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ---- Training Loop ----
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
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

    epoch_loss = running_loss / len(train_dataset)
    
    # ---- Validation ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {accuracy:.2f}%")
    
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "avengers_resnet50.pth")
        print("Model saved!")
    
    scheduler.step()

print(f"Training complete. Best Accuracy: {best_acc:.2f}%")