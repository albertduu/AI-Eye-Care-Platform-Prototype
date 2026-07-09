from pathlib import Path
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


MODEL_PATH = Path("image_model/models/oct_resnet18.pth")
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_image_model.py first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model, class_names = load_model()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]

    return {
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
        "all_probabilities": {
            class_names[i]: round(probabilities[i].item(), 4)
            for i in range(len(class_names))
        },
        "physician_review_required": True,
    }


def predict(image_path):
    return predict_image(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python image_model/predict_image.py path/to/image.jpeg")
        sys.exit(1)

    result = predict_image(sys.argv[1])
    print(json.dumps(result, indent=2))