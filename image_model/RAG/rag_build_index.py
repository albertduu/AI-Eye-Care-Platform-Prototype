import torch
import faiss
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet18
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR.parent / "models" / "oct_resnet18.pth"
DATA_DIR = BASE_DIR.parent / "data_small" / "train"
INDEX_PATH = BASE_DIR / "oct_faiss.index"
META_PATH = BASE_DIR / "oct_metadata.pkl"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("Checkpoint:", CHECKPOINT)
print("Exists:", CHECKPOINT.exists())

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
class_names = checkpoint["class_names"]

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.fc = torch.nn.Identity()
model.to(DEVICE)
model.eval()

embeddings = []
metadata = []

for class_dir in Path(DATA_DIR).iterdir():
    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for img_path in class_dir.glob("*"):
        try:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model(x).cpu().numpy()[0]

            emb = emb / np.linalg.norm(emb)

            embeddings.append(emb.astype("float32"))
            metadata.append({
                "path": str(img_path),
                "label": label
            })

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, str(INDEX_PATH).replace("\\", "/"))

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"Saved {len(metadata)} embeddings")
print(f"Classes: {class_names}")