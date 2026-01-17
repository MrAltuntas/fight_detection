"""
Video processing utilities for fight detection project.

Functions:
- extract_frames: Extract evenly-spaced frames from a video
- validate_video: Check if a video file is valid and usable
- process_all_videos: Split raw videos into train/val directories
- get_video_info: Get video metadata
"""

import os
import cv2
import shutil
import random
import numpy as np
from typing import List, Dict, Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Supported video extensions
SUPPORTED_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def extract_frames(
        video_path: str,
        num_frames: int = None,
        img_size: int = None,
        to_rgb: bool = False
) -> np.ndarray:
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (default: config.NUM_FRAMES)
        img_size: Size to resize frames to (default: config.IMG_SIZE)
        to_rgb: If True, convert BGR to RGB. If False, keep BGR (OpenCV native).

    Returns:
        numpy array of shape (num_frames, img_size, img_size, 3)
        Color format: RGB if to_rgb=True, BGR if to_rgb=False
    """
    if num_frames is None:
        num_frames = config.NUM_FRAMES
    if img_size is None:
        img_size = config.IMG_SIZE

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Calculate evenly-spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Resize frame
        frame = cv2.resize(frame, (img_size, img_size))

        # BGR → RGB conversion (optional)
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    return np.array(frames, dtype=np.uint8)

def validate_video(video_path: str, min_frames: int = 1) -> bool:
    """
    Validate that a video file is usable.

    Args:
        video_path: Path to the video file
        min_frames: Minimum number of frames required

    Returns:
        True if video is valid, False otherwise
    """
    # Check file exists
    if not os.path.exists(video_path):
        return False

    # Check extension
    if not video_path.lower().endswith(SUPPORTED_EXTENSIONS):
        return False

    # Try to open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    # Check frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames < min_frames:
        return False

    return True


def get_video_info(video_path: str) -> Dict:
    """
    Get metadata information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with keys: fps, frame_count, duration, width, height, codec

    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    duration = frame_count / fps if fps > 0 else 0

    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height,
        'codec': codec
    }


def process_all_videos(
    input_dir: str = None,
    output_dir: str = None,
    train_split: float = None,
    seed: int = 42
) -> Dict:
    """
    Process raw videos and split them into train/val directories.

    Scans input_dir/fight/ and input_dir/nonfight/ for videos,
    randomly splits them, and copies to output_dir/train/ and output_dir/val/.

    Args:
        input_dir: Directory containing fight/ and nonfight/ subdirs
                   (default: config.DATA_RAW)
        output_dir: Directory to create train/val structure
                    (default: config.DATA_PROCESSED)
        train_split: Fraction of data for training (default: config.TRAIN_SPLIT)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with statistics:
        {
            'total_videos': int,
            'fight_videos': int,
            'nonfight_videos': int,
            'train_count': int,
            'val_count': int,
            'skipped': int,
            'details': {
                'train_fight': int,
                'train_nonfight': int,
                'val_fight': int,
                'val_nonfight': int
            }
        }
    """
    if input_dir is None:
        input_dir = config.DATA_RAW
    if output_dir is None:
        output_dir = config.DATA_PROCESSED
    if train_split is None:
        train_split = config.TRAIN_SPLIT

    random.seed(seed)

    stats = {
        'total_videos': 0,
        'fight_videos': 0,
        'nonfight_videos': 0,
        'train_count': 0,
        'val_count': 0,
        'skipped': 0,
        'details': {
            'train_fight': 0,
            'train_nonfight': 0,
            'val_fight': 0,
            'val_nonfight': 0
        }
    }

    # Create output directory structure
    for split in ['train', 'val']:
        for class_name in config.CLASSES:
            dir_path = os.path.join(output_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)

    # Process each class
    for class_name in config.CLASSES:
        class_input_dir = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_input_dir):
            print(f"Warning: Directory not found: {class_input_dir}")
            continue

        # Get all video files
        video_files = [
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]

        # Validate and filter videos
        valid_videos = []
        for video_file in video_files:
            video_path = os.path.join(class_input_dir, video_file)
            if validate_video(video_path):
                valid_videos.append(video_file)
            else:
                stats['skipped'] += 1
                print(f"Skipping invalid video: {video_path}")

        # Update class stats
        if class_name == 'fight':
            stats['fight_videos'] = len(valid_videos)
        else:
            stats['nonfight_videos'] = len(valid_videos)

        stats['total_videos'] += len(valid_videos)

        # Shuffle and split
        random.shuffle(valid_videos)
        split_idx = int(len(valid_videos) * train_split)

        train_videos = valid_videos[:split_idx]
        val_videos = valid_videos[split_idx:]

        # Copy train videos
        for video_file in train_videos:
            src = os.path.join(class_input_dir, video_file)
            dst = os.path.join(output_dir, 'train', class_name, video_file)
            shutil.copy2(src, dst)
            stats['train_count'] += 1
            stats['details'][f'train_{class_name}'] += 1

        # Copy val videos
        for video_file in val_videos:
            src = os.path.join(class_input_dir, video_file)
            dst = os.path.join(output_dir, 'val', class_name, video_file)
            shutil.copy2(src, dst)
            stats['val_count'] += 1
            stats['details'][f'val_{class_name}'] += 1

    # Print summary
    print("\n" + "=" * 50)
    print("Video Processing Complete")
    print("=" * 50)
    print(f"Total videos processed: {stats['total_videos']}")
    print(f"  - Fight videos: {stats['fight_videos']}")
    print(f"  - Non-fight videos: {stats['nonfight_videos']}")
    print(f"  - Skipped (invalid): {stats['skipped']}")
    print(f"\nTrain/Val Split ({train_split:.0%}/{1-train_split:.0%}):")
    print(f"  - Train: {stats['train_count']} videos")
    print(f"    - Fight: {stats['details']['train_fight']}")
    print(f"    - Non-fight: {stats['details']['train_nonfight']}")
    print(f"  - Val: {stats['val_count']} videos")
    print(f"    - Fight: {stats['details']['val_fight']}")
    print(f"    - Non-fight: {stats['details']['val_nonfight']}")
    print("=" * 50)

    return stats


if __name__ == "__main__":
    process_all_videos()
