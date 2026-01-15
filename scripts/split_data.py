# scripts/01_split_data.py

import os
import shutil
from sklearn.model_selection import train_test_split

BASE_DIR = "/home/jayanynunna/Kuchipudi-Mudra-Dataset-master"  # original dataset
OUT_DIR = "/home/jayanynunna/mudra_dataset_2"   # train, val, test split will go here
SPLITS = (0.7, 0.1, 0.2)

def split_dataset(base_dir, output_dir, splits):
    train_pct, val_pct, test_pct = splits

    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_path):
            continue

        images = [img for img in os.listdir(label_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train, temp = train_test_split(images, train_size=train_pct, random_state=42)
        val, teCreate a new folder called background_invariant_mudrast = train_test_split(temp, test_size=test_pct / (val_pct + test_pct), random_state=42)

        for split_name, split_images in zip(['train', 'val', 'test'], [train, val, test]):
            split_dir = os.path.join(output_dir, split_name, label)
            os.makedirs(split_dir, exist_ok=True)

            for img in split_images:
                src = os.path.join(label_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)

    print(f"✅ Dataset split completed. Output saved in: {output_dir}")

if __name__ == "__main__":
    split_dataset(BASE_DIR, OUT_DIR, SPLITS)
