import os

SEED = 42


# Dataset
NUM_WORKERS = 4

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_RAW = os.path.join(PROJECT_ROOT, "data/raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data/processed")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "checkpoints/best_model.pth")

# Model params
NUM_FRAMES = 16
IMG_SIZE = 224
LSTM_HIDDEN = 512
NUM_CLASSES = 2
MOBILENET_FEATURES = 1280  # MobileNetV2 output features

# Training params
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
PATIENCE = 10

# Class labels
CLASSES = ["nonfight", "fight"]

# Kaggle API Token
KAGGLE_API_TOKEN = "KGAT_7f803a846a569fe86806eca48e316343"
