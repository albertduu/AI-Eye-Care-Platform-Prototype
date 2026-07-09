from pathlib import Path
import shutil
import random

SOURCE = Path("image_model/data/train")
DEST = Path("image_model/data_small/train")

IMAGES_PER_CLASS = 1000

random.seed(42)

DEST.mkdir(parents=True, exist_ok=True)

for class_dir in SOURCE.iterdir():
    if not class_dir.is_dir():
        continue

    dest_class = DEST / class_dir.name
    dest_class.mkdir(parents=True, exist_ok=True)

    images = list(class_dir.glob("*"))
    random.shuffle(images)

    selected = images[:IMAGES_PER_CLASS]

    print(f"{class_dir.name}: copying {len(selected)} images")

    for img in selected:
        shutil.copy2(img, dest_class / img.name)

print("Done!")