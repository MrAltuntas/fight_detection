"""
Dataset Download Script for Real Life Violence Situations Dataset
Dataset: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

This script downloads and organizes the dataset into the proper folder structure.
Requires: Kaggle API credentials (~/.kaggle/kaggle.json)
"""

import os
import sys
import shutil
from pathlib import Path
import config

DATASET_NAME = "mohamedmustafa/real-life-violence-situations-dataset"


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured."""
    if hasattr(config, 'KAGGLE_API_TOKEN') and config.KAGGLE_API_TOKEN:
        print(f"Using Kaggle API Token from config.py")
        return True

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: Kaggle credentials not found!")
        print(f"Expected location: {kaggle_json}")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' under API section")
        print("3. Move downloaded kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_dataset():
    """Download dataset using Kaggle API."""
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub package not installed!")
        print("Run: pip install kagglehub")
        return None

    print(f"Downloading dataset: {DATASET_NAME}")
    print("This may take a few minutes...\n")

    # Set Kaggle API token from config
    os.environ['KAGGLE_API_TOKEN'] = config.KAGGLE_API_TOKEN

    # Download dataset using kagglehub
    download_path = kagglehub.dataset_download(DATASET_NAME)

    print(f"Dataset downloaded to: {download_path}")

    return Path(download_path)


def organize_dataset(download_path):
    """Organize dataset into fight/nonfight folders."""
    print(f"Organizing dataset from: {download_path}")

    # Find Violence and NonViolence folders
    # Dataset structure: Real Life Violence Dataset/Violence/*.avi
    #                   Real Life Violence Dataset/NonViolence/*.avi
    violence_dir = None
    nonviolence_dir = None

    for root, dirs, files in os.walk(download_path):
        root_path = Path(root)
        if root_path.name == "Violence" and "NonViolence" not in str(root_path):
            violence_dir = root_path
        elif root_path.name == "NonViolence":
            nonviolence_dir = root_path

    if not violence_dir or not nonviolence_dir:
        # Try alternative naming
        for root, dirs, files in os.walk(download_path):
            for d in dirs:
                if d.lower() == "violence":
                    violence_dir = Path(root) / d
                elif d.lower() == "nonviolence":
                    nonviolence_dir = Path(root) / d

    if not violence_dir or not nonviolence_dir:
        print("ERROR: Could not find Violence/NonViolence folders in dataset")
        print("Downloaded contents:")
        for item in Path(download_path).rglob("*"):
            print(f"  {item}")
        return False

    print(f"\nFound Violence folder: {violence_dir}")
    print(f"Found NonViolence folder: {nonviolence_dir}")

    # Copy files to proper locations
    data_raw = Path(config.DATA_RAW)
    fight_dest = data_raw / "fight"
    nonfight_dest = data_raw / "nonfight"

    # Create destination folders if they don't exist
    fight_dest.mkdir(parents=True, exist_ok=True)
    nonfight_dest.mkdir(parents=True, exist_ok=True)

    # Clear destination folders
    for dest in [fight_dest, nonfight_dest]:
        for f in dest.iterdir():
            if f.is_file():
                f.unlink()

    # Copy Violence videos to fight/
    fight_count = 0
    for video_file in violence_dir.iterdir():
        if video_file.is_file():
            shutil.copy2(str(video_file), str(fight_dest / video_file.name))
            fight_count += 1

    # Copy NonViolence videos to nonfight/
    nonfight_count = 0
    for video_file in nonviolence_dir.iterdir():
        if video_file.is_file():
            shutil.copy2(str(video_file), str(nonfight_dest / video_file.name))
            nonfight_count += 1

    print(f"\nCopied {fight_count} videos to {fight_dest}")
    print(f"Copied {nonfight_count} videos to {nonfight_dest}")

    return True


def main():
    print("=" * 60)
    print("Real Life Violence Situations Dataset Downloader")
    print("=" * 60 + "\n")

    # Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)

    # Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        sys.exit(1)

    # Organize dataset
    success = organize_dataset(dataset_path)
    if not success:
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)

    data_raw = Path(config.DATA_RAW)
    fight_videos = list((data_raw / "fight").glob("*"))
    nonfight_videos = list((data_raw / "nonfight").glob("*"))

    print(f"\nDataset location: {data_raw}")
    print(f"  - Fight videos: {len(fight_videos)}")
    print(f"  - Non-fight videos: {len(nonfight_videos)}")
    print(f"  - Total: {len(fight_videos) + len(nonfight_videos)}")


if __name__ == "__main__":
    main()
