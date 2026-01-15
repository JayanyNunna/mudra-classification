# Dataset

Place the Kuchipudi Mudra Dataset here.

## Setup

1. Extract `Kuchipudi-Mudra-Dataset-master.zip`
2. Run `python scripts/split_data.py` to create train/val/test splits

## Structure After Split
```
data/
├── raw/          # Original dataset
├── train/        # 70% training data
├── val/          # 10% validation data
└── test/         # 20% test data
```
