#!/usr/bin/env python3
"""
Single video inference script for violence detection.

Usage:
    python predict.py --video path/to/video.mp4
    python predict.py --video path/to/video.mp4 --model checkpoints/best_model.pth
"""

import os
import argparse
import torch
from torchvision import transforms

import config
from models import ViolenceDetector
from utils.video_processor import extract_frames
from utils.helpers import get_device, load_checkpoint


def predict_video(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose
) -> tuple:
    """
    Run inference on a single video.

    Args:
        video_path: Path to video file
        model: Trained model
        device: Device to run inference on
        transform: Image transforms

    Returns:
        tuple: (predicted_class, class_name, confidence)
    """
    # Extract frames from video
    frames = extract_frames(video_path, to_rgb=True)

    # Apply transforms to each frame
    transformed_frames = []
    for frame in frames:
        frame_tensor = transform(frame)
        transformed_frames.append(frame_tensor)

    # Stack frames: (num_frames, 3, img_size, img_size)
    frames_tensor = torch.stack(transformed_frames, dim=0)

    # Add batch dimension: (1, num_frames, 3, img_size, img_size)
    frames_tensor = frames_tensor.unsqueeze(0).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(frames_tensor)
        probs = torch.softmax(outputs, dim=1)

    # Get prediction
    confidence, predicted_class = torch.max(probs, dim=1)
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    class_name = config.CLASSES[predicted_class]

    return predicted_class, class_name, confidence


def main():
    parser = argparse.ArgumentParser(description='Predict violence in a video')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--model', type=str, default=config.MODEL_SAVE_PATH,
                        help=f'Path to model checkpoint (default: {config.MODEL_SAVE_PATH})')
    args = parser.parse_args()

    # Check video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        print("Please train the model first using train.py")
        return

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.model}")
    model = ViolenceDetector(
        num_frames=config.NUM_FRAMES,
        num_classes=config.NUM_CLASSES,
        lstm_hidden=config.LSTM_HIDDEN
    )
    model = model.to(device)
    load_checkpoint(model, args.model, device=device)

    # Define transform (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Run prediction
    print(f"\nAnalyzing video: {args.video}")
    predicted_class, class_name, confidence = predict_video(
        args.video, model, device, transform
    )

    # Display result
    print("\n" + "=" * 40)
    print("PREDICTION RESULT")
    print("=" * 40)

    if class_name == "fight":
        result_text = "Fight Detected"
    else:
        result_text = "Non-fight (Normal)"

    print(f"  Result:     {result_text}")
    print(f"  Class:      {class_name}")
    print(f"  Confidence: {confidence * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
