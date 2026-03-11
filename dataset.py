import os
import random
import shutil
from pathlib import Path

TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")

TEST_RATIO = 0.2


def split_dataset():

    classes = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]

    for cls in classes:

        train_class_path = TRAIN_DIR / cls.name
        test_class_path = TEST_DIR / cls.name

        test_class_path.mkdir(parents=True, exist_ok=True)

        images = list(train_class_path.glob("*"))

        random.shuffle(images)

        split_size = int(len(images) * TEST_RATIO)

        test_images = images[:split_size]

        print(f"\nClass: {cls.name}")
        print(f"Total images: {len(images)}")
        print(f"Moving {split_size} images to test set")

        for img in test_images:
            dest = test_class_path / img.name
            shutil.move(str(img), str(dest))


if __name__ == "__main__":
    split_dataset()