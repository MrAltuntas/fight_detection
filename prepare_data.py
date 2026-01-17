#!/usr/bin/env python3
"""
Data preparation pipeline - run all steps with a single command.

Usage:
    python prepare_data.py
"""

import os
import config
from utils.download_dataset import download_dataset, organize_dataset, check_kaggle_credentials
from utils.video_processor import process_all_videos
from utils.preprocess_frames import main as preprocess_tensors
from data.dataset import ViolenceDataset


def check_raw_dataset_exists():
    """Check if raw dataset already exists with videos."""
    fight_dir = os.path.join(config.DATA_RAW, "fight")
    nonfight_dir = os.path.join(config.DATA_RAW, "nonfight")
    video_ext = ('.mp4', '.avi', '.mov', '.mkv')

    fight_count = len([f for f in os.listdir(fight_dir) if f.lower().endswith(video_ext)]) if os.path.isdir(fight_dir) else 0
    nonfight_count = len([f for f in os.listdir(nonfight_dir) if f.lower().endswith(video_ext)]) if os.path.isdir(nonfight_dir) else 0

    return fight_count > 0 and nonfight_count > 0


def verify_dataset():
    """Verify the prepared dataset is usable."""
    train_dir = os.path.join(config.DATA_PROCESSED, "train")
    val_dir = os.path.join(config.DATA_PROCESSED, "val")

    train_dataset = ViolenceDataset(train_dir)
    val_dataset = ViolenceDataset(val_dir)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    if len(train_dataset) > 0:
        frames, label = train_dataset[0]
        expected_shape = (config.NUM_FRAMES, 3, config.IMG_SIZE, config.IMG_SIZE)
        assert frames.shape == expected_shape, f"Shape mismatch: {frames.shape}"
        print(f"  Sample shape: {frames.shape} OK")

    return len(train_dataset) > 0 and len(val_dataset) > 0


def main():
    print("=" * 50)
    print("DATA PREPARATION PIPELINE")
    print("=" * 50)

    # Step 1: Dataset download
    print("\n[1/4] Checking raw dataset...")
    if check_raw_dataset_exists():
        print("  Dataset exists, skipping download.")
    else:
        print("  Downloading from Kaggle...")
        if not check_kaggle_credentials():
            return
        download_path = download_dataset()
        if not download_path or not organize_dataset(download_path):
            print("ERROR: Download failed!")
            return

    # Step 2: Video processing (train/val split)
    print("\n[2/4] Processing videos (train/val split)...")
    stats = process_all_videos()
    if stats['total_videos'] == 0:
        print("ERROR: No videos found!")
        return

    # Step 3: Tensor preprocessing
    print("\n[3/4] Preprocessing tensors...")
    preprocess_tensors()

    # Step 4: Verify dataset
    print("\n[4/4] Verifying dataset...")
    if not verify_dataset():
        print("ERROR: Dataset verification failed!")
        return

    print("\n" + "=" * 50)
    print("Done! Run 'python train.py' to start training.")
    print("=" * 50)


if __name__ == "__main__":
    main()
