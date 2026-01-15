import os, shutil
from sklearn.model_selection import train_test_split

base = "/home/jayanynunna/Kuchipudi-Mudra-Dataset-master"
out = "/home/jayanynunna/mudra-dataset"  # will contain train, val, test

def split_dataset(base, out, splits=(0.7, 0.1, 0.2)):
    train_pct, val_pct, test_pct = splits
    for label in os.listdir(base):
        label_path = os.path.join(base, label)
        if not os.path.isdir(label_path):
            continue  # Skip files like LICENSE.md
        imgs = os.listdir(label_path)
        train, temp = train_test_split(imgs, train_size=train_pct, random_state=42)
        val, test = train_test_split(temp, test_size=test_pct / (val_pct + test_pct), random_state=42)
        for folder, group in zip(['train','val','test'], [train, val, test]):
            ld = os.path.join(out, folder, label)
            os.makedirs(ld, exist_ok=True)
            for img in group:
                shutil.copy(os.path.join(base, label, img), os.path.join(ld, img))

if __name__ == "__main__":
    split_dataset(base, out)
