
import os
import copy
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

# ==============================
# SETTINGS
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "component_classifier_v2.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# IMAGE TRANSFORMS
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# LOAD DATASET
# ==============================
if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

# Separate dataset objects let training and validation
# use different image transforms.
train_dataset_full = datasets.ImageFolder(
    DATASET_DIR, transform=train_transform
)
eval_dataset_full = datasets.ImageFolder(
    DATASET_DIR, transform=eval_transform
)

class_names = train_dataset_full.classes
num_classes = len(class_names)

if num_classes != 5:
    raise ValueError(
        f"Expected 5 class folders, found {num_classes}: "
        f"{class_names}"
    )

print("\nClasses:", class_names)

for class_name, count in zip(
    train_dataset_full.classes,
    np.bincount(train_dataset_full.targets)
):
    print(f"{class_name}: {count} images")

# Make a reproducible, stratified 80/20 split.
targets = np.array(train_dataset_full.targets)
train_indices = []
val_indices = []

rng = np.random.default_rng(SEED)

for class_id in range(num_classes):
    indices = np.where(targets == class_id)[0]
    rng.shuffle(indices)

    if len(indices) < 5:
        raise ValueError(
            f"Class '{class_names[class_id]}' needs "
            "at least 5 images."
        )

    val_count = max(1, round(len(indices) * 0.2))
    val_indices.extend(indices[:val_count].tolist())
    train_indices.extend(indices[val_count:].tolist())

train_data = Subset(train_dataset_full, train_indices)
val_data = Subset(eval_dataset_full, val_indices)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"\nTraining images: {len(train_data)}")
print(f"Validation images: {len(val_data)}")

# ==============================
# LOAD PRETRAINED MODEL
# ==============================
print("\nLoading pretrained MobileNetV3-Small...")
print("The first run may download model weights.")

weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Freeze the pretrained feature extractor first.
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace the final classification layer for our 5 classes.
input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# ==============================
# TRAIN AND VALIDATE
# ==============================
best_accuracy = -1.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    train_loss_total = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * images.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

    train_loss = train_loss_total / len(train_data)
    train_accuracy = train_correct / len(train_data)

    model.eval()
    val_loss_total = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

    val_loss = val_loss_total / len(val_data)
    val_accuracy = val_correct / len(val_data)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy:.1%} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_accuracy:.1%}"
    )

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_weights = copy.deepcopy(model.state_dict())

# ==============================
# SAVE BEST MODEL
# ==============================
model.load_state_dict(best_weights)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "image_size": IMAGE_SIZE,
    "architecture": "mobilenet_v3_small",
    "validation_accuracy": best_accuracy
}, MODEL_PATH)

print("\nTraining complete!")
print(f"Best validation accuracy: {best_accuracy:.1%}")
print(f"Model saved to: {MODEL_PATH}")
print(
    "\nNext: test the model on new images before using "
    "predictions to change inventory quantities."
)
