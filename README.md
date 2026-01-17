# Fight Detection

A deep learning-based violence detection system that identifies fight/violence in videos using a CNN+LSTM architecture. The model uses MobileNetV2 as a feature extractor combined with LSTM for temporal sequence modeling.

## Project Overview

This project implements a binary video classifier that can detect violent/fight situations in video footage. The architecture combines:
- **MobileNetV2** (pretrained on ImageNet) for spatial feature extraction from video frames
- **LSTM** for temporal sequence modeling across frames
- Achieves real-time inference on standard hardware

## Architecture

- **Input**: Video files (MP4, AVI, MOV, MKV)
- **Frame Extraction**: 16 evenly-spaced frames per video
- **Image Size**: 224x224 pixels
- **Feature Extractor**: MobileNetV2 (frozen CNN layers)
- **Temporal Model**: Single-layer LSTM (512 hidden units)
- **Output**: Binary classification (fight/nonfight)

## Model Details

### ViolenceDetector Architecture

```
Input: (batch, 16, 3, 224, 224)
    ↓
MobileNetV2 Feature Extractor (frozen)
    ↓
Global Average Pooling
    ↓
Reshape: (batch, 16, 1280)
    ↓
LSTM (512 hidden units)
    ↓
Dropout (0.5)
    ↓
Linear Classifier (512 → 2)
    ↓
Output: (batch, 2) [nonfight, fight]
```


**Key Features:**
- **Transfer Learning**: Pretrained MobileNetV2 weights (frozen during training)
- **Efficient**: Only LSTM and classifier layers are trainable
- **Temporal Modeling**: LSTM captures motion patterns across frames
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Full pipeline (download + preprocess + train)
python prepare_data.py
python train.py
```

## Run Individually

```bash
# 1. Download dataset (Kaggle API required)
python utils/download_dataset.py

# 2. Video processing and train/val split
python utils/video_processor.py

# 3. Training
python train.py
python train.py --resume checkpoints/best_model.pth  # resume

# 4. Test
python test.py

# 5. Predict
python predict.py --video video.mp4
```

## Folder Structure

```
data/
├── raw/          # Raw videos (fight/, nonfight/)
├── processed/    # Train/val split
└── tensors/      # Preprocessed tensors

checkpoints/      # Model and plots
```

## Configuration

Edit `config.py` to modify hyperparameters:

```python
# Model parameters
NUM_FRAMES = 16          # Frames per video
IMG_SIZE = 224           # Image size (224x224)
LSTM_HIDDEN = 512        # LSTM hidden units

# Training parameters
BATCH_SIZE = 8           # Batch size
EPOCHS = 30              # Max epochs
LEARNING_RATE = 0.001    # Initial learning rate
TRAIN_SPLIT = 0.8        # Train/val split ratio
PATIENCE = 10            # Early stopping patience

# Paths
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
MODEL_SAVE_PATH = "checkpoints/best_model.pth"
```

## Dataset

[Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) - Kaggle