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

## Requirements

- Python 3.12
- CUDA-capable GPU (optional, but recommended for training)
- Kaggle API credentials for dataset download

## Installation

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The required packages include:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- tqdm >= 4.65.0
- kaggle >= 1.5.0

## Dataset Setup

This project uses the [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) from Kaggle.

### Step 1: Configure Kaggle API

This will:
- Download the dataset from Kaggle
- Organize videos into `data/raw/fight/` and `data/raw/nonfight/` folders
- Display download statistics

Expected output structure:
```
data/raw/
├── fight/       # Violence videos
└── nonfight/    # Non-violence videos
```

### Step 3: Preprocess and Split Dataset

Process the raw videos into train/validation splits:
```bash
python -m utils.video_processor --process
```

This will:
- Validate all video files
- Split data into 80% training, 20% validation
- Copy videos to `data/processed/train/` and `data/processed/val/`
- Display processing statistics

Expected output structure:
```
data/processed/
├── train/
│   ├── fight/
│   └── nonfight/
└── val/
    ├── fight/
    └── nonfight/
```

## Training

Train the model with default parameters:
```bash
python train.py
```

### Training Options

```bash
# Resume training from a checkpoint
python train.py --resume checkpoints/best_model.pth
```

### Training Outputs

Training will generate:
- `checkpoints/best_model.pth` - Best model checkpoint
- `checkpoints/training_curves.png` - Loss and accuracy plots

## Evaluation

Evaluate the trained model on validation set:
```bash
python test.py
```

### Evaluation Options

```bash
# Evaluate on training set
python test.py --split train

# Use custom model checkpoint
python test.py --model checkpoints/best_model.pth

# Custom batch size
python test.py --batch_size 16
```

### Evaluation Outputs

The script will generate:
- Accuracy, Precision, Recall, F1-Score metrics
- Per-class classification report
- Confusion matrix (printed and saved as PNG)
- `checkpoints/confusion_matrix.png` - Visualization

## Inference

Run inference on a single video:
```bash
python predict.py --video path/to/video.mp4
```

### Prediction Options

```bash
# Use custom model checkpoint
python predict.py --video path/to/video.mp4 --model checkpoints/best_model.pth
```


## Project Structure

```
fight_detection/
├── config.py                 # Configuration and hyperparameters
├── download_dataset.py       # Dataset download script
├── train.py                  # Training script
├── test.py                   # Evaluation script
├── predict.py                # Single video inference
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── raw/                  # Raw downloaded videos
│   │   ├── fight/
│   │   └── nonfight/
│   └── processed/            # Preprocessed train/val splits
│       ├── train/
│       └── val/
│
├── models/
│   ├── __init__.py
│   └── model.py              # ViolenceDetector model definition
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py            # Training utilities
│   └── video_processor.py   # Video processing utilities
│
└── checkpoints/              # Saved models and visualizations
    ├── best_model.pth
    ├── training_curves.png
    └── confusion_matrix.png
```

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

### Key Features

- **Transfer Learning**: Pretrained MobileNetV2 weights (frozen during training)
- **Efficient**: Only LSTM and classifier layers are trainable
- **Temporal Modeling**: LSTM captures motion patterns across frames
- **Data Augmentation**: ImageNet normalization applied
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)

## Usage Examples

### Complete Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and organize dataset
python download_dataset.py

# 3. Preprocess videos into train/val splits
python -m utils.video_processor --process

# 4. Train the model
python train.py

# 5. Evaluate on validation set
python test.py

# 6. Run inference on new video
python predict.py --video test_video.mp4
```

### Video Processing Utilities

```bash
# Get video information
python -m utils.video_processor --info path/to/video.mp4

# Validate video file
python -m utils.video_processor --validate path/to/video.mp4

# Extract frames for testing
python -m utils.video_processor --extract path/to/video.mp4
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

## Performance

The model achieves competitive performance on the Real Life Violence Situations Dataset with efficient inference suitable for real-time applications.

Training time (approximate):
- ~5-10 minutes per epoch on modern GPU
- ~20-30 epochs for convergence (with early stopping)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `config.py`
2. **Video loading errors**: Ensure videos are in supported formats (MP4, AVI, MOV, MKV)
3. **Kaggle download fails**: Check Kaggle API credentials
4. **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset: [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- Pretrained Model: MobileNetV2 from torchvision
- Framework: PyTorch
