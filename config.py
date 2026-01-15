# config.py
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FEATURE_DIR = os.path.join(PROJECT_ROOT, "features")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Dataset settings
BASE_DIR = os.getenv("BASE_DIR", "/home/jayanynunna/Kuchipudi-Mudra-Dataset-master")
NUM_CLASSES = 28

# Data split
TRAIN_PCT = 0.7
VAL_PCT = 0.1
TEST_PCT = 0.2

# Model parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SVM_KERNEL = 'rbf'
SVM_C = 10
SVM_GAMMA = 'scale'

# Random seed
RANDOM_STATE = 42
