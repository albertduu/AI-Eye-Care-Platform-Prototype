from pathlib import Path
import shutil
import random

SOURCE = Path("data/train")          # run this from inside image_model
DEST = Path("data_small")

TRAIN_PER_CLASS = 1000
VAL_PER_CLASS = 250
TEST_PER_CLASS = 250

random.seed(42)

for split in ["train", "val", "test"]:
    (DEST / split).mkdir(parents=True, exist_ok=True)

for class_dir in SOURCE.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*"))
    random.shuffle(images)

    train_images = images[:TRAIN_PER_CLASS]
    val_images = images[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
    test_images = images[TRAIN_PER_CLASS + VAL_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS]

    print(f"{class_dir.name}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]:
        dest_class = DEST / split_name / class_dir.name
        dest_class.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(img, dest_class / img.name)

print("Done creating small train/val/test dataset.")