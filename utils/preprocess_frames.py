#!/usr/bin/env python3
"""
Preprocess video frames and save as tensors for faster training.

This script extracts frames from all videos in data/processed/ and saves them
as pre-processed tensors in data/tensors/. This eliminates video I/O during
training, significantly improving GPU utilization.

Usage:
    python utils/preprocess_frames.py
"""

import os
import sys
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.video_processor import extract_frames

# Supported video extensions
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def get_transform():
    """Get the same transform used during training."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def process_video_to_tensor(video_path: str, transform) -> torch.Tensor:
    """
    Extract frames from video and convert to tensor.

    Args:
        video_path: Path to video file
        transform: Torchvision transform to apply

    Returns:
        Tensor of shape (num_frames, 3, img_size, img_size)
    """
    # Extract frames (already resized)
    frames = extract_frames(video_path, to_rgb=True)

    # Apply transform to each frame and stack
    transformed_frames = []
    for frame in frames:
        frame_tensor = transform(frame)
        transformed_frames.append(frame_tensor)

    return torch.stack(transformed_frames, dim=0)


def preprocess_split(split: str, transform) -> dict:
    """
    Preprocess all videos in a split (train or val).

    Args:
        split: 'train' or 'val'
        transform: Torchvision transform to apply

    Returns:
        Statistics dictionary
    """
    input_dir = os.path.join(config.DATA_PROCESSED, split)
    output_dir = os.path.join(config.DATA_TENSORS, split)

    stats = {'processed': 0, 'skipped': 0, 'errors': 0}

    for class_name in config.CLASSES:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_input_dir):
            print(f"  Warning: Directory not found: {class_input_dir}")
            continue

        # Create output directory
        os.makedirs(class_output_dir, exist_ok=True)

        # Get all video files
        video_files = [
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]

        # Process each video
        for video_file in tqdm(video_files, desc=f"  {class_name}", leave=False):
            video_path = os.path.join(class_input_dir, video_file)

            # Output path: same name but .pt extension
            tensor_name = os.path.splitext(video_file)[0] + '.pt'
            tensor_path = os.path.join(class_output_dir, tensor_name)

            # Skip if already processed
            if os.path.exists(tensor_path):
                stats['skipped'] += 1
                continue

            try:
                # Process video and save tensor
                tensor = process_video_to_tensor(video_path, transform)
                torch.save(tensor, tensor_path)
                stats['processed'] += 1
            except Exception as e:
                print(f"  Error processing {video_file}: {e}")
                stats['errors'] += 1

    return stats


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("Frame Preprocessing for GPU Optimization")
    print("=" * 60)
    print(f"Source: {config.DATA_PROCESSED}")
    print(f"Output: {config.DATA_TENSORS}")
    print(f"Frame shape: ({config.NUM_FRAMES}, 3, {config.IMG_SIZE}, {config.IMG_SIZE})")
    print()

    # Create transform
    transform = get_transform()

    total_stats = {'processed': 0, 'skipped': 0, 'errors': 0}

    # Process train and val splits
    for split in ['train', 'val']:
        print(f"Processing {split} split...")
        stats = preprocess_split(split, transform)

        for key in total_stats:
            total_stats[key] += stats[key]

        print(f"  Processed: {stats['processed']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")

    print()
    print("=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped (already exists): {total_stats['skipped']}")
    print(f"Total errors: {total_stats['errors']}")

if __name__ == "__main__":
    main()
