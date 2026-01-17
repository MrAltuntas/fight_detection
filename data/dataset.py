import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.video_processor import extract_frames


class ViolenceDataset(Dataset):
    """PyTorch Dataset for violence detection videos."""

    def __init__(self, root_dir, transform=None, tensor_dir=None):
        """
        Args:
            root_dir: Path to train/ or val/ folder (contains fight/ and nonfight/)
            transform: Optional torchvision transforms. If None, uses default.
            tensor_dir: Path to tensor directory. If None, auto-detects from config.
        """
        self.root_dir = root_dir
        self.samples = []  # List of (path, label, is_tensor)

        # Determine tensor directory (parallel structure to root_dir)
        if tensor_dir is None:
            # Extract split name (train/val) from root_dir
            split_name = os.path.basename(root_dir)
            self.tensor_dir = os.path.join(config.DATA_TENSORS, split_name)
        else:
            self.tensor_dir = tensor_dir

        # Default transform: convert to tensor and normalize with ImageNet stats
        # Note: Resize is handled by extract_frames() for efficiency
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        # Scan subdirectories for videos/tensors
        # label: 0=nonfight, 1=fight (matching config.CLASSES order)
        for label_idx, class_name in enumerate(config.CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            tensor_class_dir = os.path.join(self.tensor_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            for video_file in os.listdir(class_dir):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_dir, video_file)

                    # Check if pre-processed tensor exists
                    tensor_name = os.path.splitext(video_file)[0] + '.pt'
                    tensor_path = os.path.join(tensor_class_dir, tensor_name)

                    if os.path.exists(tensor_path):
                        # Use tensor (faster)
                        self.samples.append((tensor_path, label_idx, True))
                    else:
                        # Fallback to video
                        self.samples.append((video_path, label_idx, False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, is_tensor = self.samples[idx]

        if is_tensor:
            # Load pre-processed tensor (fast path)
            frames_tensor = torch.load(path, weights_only=True)
        else:
            # Fallback: Load frames from video (slow path)
            frames = extract_frames(path, to_rgb=True)

            # Apply transforms to each frame and stack
            transformed_frames = []
            for frame in frames:
                frame_tensor = self.transform(frame)
                transformed_frames.append(frame_tensor)

            # Stack frames: (num_frames, 3, img_size, img_size)
            frames_tensor = torch.stack(transformed_frames, dim=0)

        return frames_tensor, label

def get_dataloaders(train_dir=None, val_dir=None, batch_size=None, num_workers=4):
    """
    Create train and validation DataLoaders.

    Args:
        train_dir: Path to training data directory (default: config.DATA_PROCESSED/train)
        val_dir: Path to validation data directory (default: config.DATA_PROCESSED/val)
        batch_size: Batch size (defaults to config.BATCH_SIZE)
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, val_loader
    """
    if train_dir is None:
        train_dir = os.path.join(config.DATA_PROCESSED, 'train')
    if val_dir is None:
        val_dir = os.path.join(config.DATA_PROCESSED, 'val')
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_dataset = ViolenceDataset(train_dir)
    val_dataset = ViolenceDataset(val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Verification test
    print("Testing ViolenceDataset...")

    # Use processed data directory for testing
    test_dir = config.DATA_PROCESSED

    # Check if train directory exists
    train_dir = os.path.join(test_dir, "train")
    if os.path.exists(train_dir):
        dataset = ViolenceDataset(train_dir)
        print(f"Dataset length: {len(dataset)}")

        if len(dataset) > 0:
            # Get single sample
            frames, label = dataset[0]
            print(f"Sample shape: {frames.shape}")
            print(f"Expected shape: ({config.NUM_FRAMES}, 3, {config.IMG_SIZE}, {config.IMG_SIZE})")
            print(f"Label: {label} ({config.CLASSES[label]})")

            # Verify shape
            expected_shape = (config.NUM_FRAMES, 3, config.IMG_SIZE, config.IMG_SIZE)
            assert frames.shape == expected_shape, f"Shape mismatch: {frames.shape} vs {expected_shape}"
            assert label in [0, 1], f"Invalid label: {label}"

            print("Single sample verification PASSED!")

            # Test DataLoader
            val_dir = os.path.join(test_dir, "val")
            if os.path.exists(val_dir):
                train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size=2, num_workers=0)
                batch_frames, batch_labels = next(iter(train_loader))
                print(f"Batch shape: {batch_frames.shape}")
                print("DataLoader verification PASSED!")
        else:
            print("No videos found in dataset directory.")
    else:
        print(f"Test directory not found: {train_dir}")
        print("Run data preparation first to create train/val splits.")
