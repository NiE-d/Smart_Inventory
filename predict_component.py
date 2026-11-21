
from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "component_classifier_v2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

model = models.mobilenet_v3_small(weights=None)
input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, len(class_names))

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    scores, indices = torch.topk(probabilities, k=len(class_names))

    print("\nPredictions:")
    for score, index in zip(scores.cpu(), indices.cpu()):
        print(f"{class_names[index.item()]}: {score.item():.1%}")

    best_index = indices[0].item()
    print(f"\nPredicted component: {class_names[best_index]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: py predict_component.py "path\\to\\image.jpg"')
        sys.exit(1)

    predict(sys.argv[1])
