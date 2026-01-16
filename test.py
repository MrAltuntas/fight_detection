#!/usr/bin/env python3
"""
Evaluation script for violence detection model.

Usage:
    python test.py
    python test.py --model checkpoints/best_model.pth
    python test.py --split val
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt

import config
from models import ViolenceDetector
from data.dataset import ViolenceDataset
from torch.utils.data import DataLoader
from utils.helpers import get_device, load_checkpoint


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate model on dataset.

    Returns:
        tuple: (all_predictions, all_labels, all_probs)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for frames, labels in tqdm(data_loader, desc='Evaluating'):
            frames = frames.to(device)

            outputs = model(frames)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, dim=1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probs)
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str
) -> None:
    """
    Plot and save confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix'
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate violence detection model')
    parser.add_argument('--model', type=str, default=config.MODEL_SAVE_PATH,
                        help=f'Path to model checkpoint (default: {config.MODEL_SAVE_PATH})')
    parser.add_argument('--data_dir', type=str, default=config.DATA_PROCESSED,
                        help=f'Path to data directory (default: {config.DATA_PROCESSED})')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to evaluate (default: val)')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        print("Please train the model first using train.py")
        return

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    data_path = os.path.join(args.data_dir, args.split)
    if not os.path.exists(data_path):
        print(f"Error: Data directory not found: {data_path}")
        return

    print(f"Loading {args.split} dataset from: {data_path}")
    dataset = ViolenceDataset(data_path)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Load model
    print(f"Loading model from: {args.model}")
    model = ViolenceDetector(
        num_frames=config.NUM_FRAMES,
        num_classes=config.NUM_CLASSES,
        lstm_hidden=config.LSTM_HIDDEN
    )
    model = model.to(device)

    metadata = load_checkpoint(model, args.model, device=device)
    print(f"Loaded checkpoint from epoch {metadata['epoch']}")
    print(f"Checkpoint val_accuracy: {metadata['val_accuracy']:.4f}")

    # Evaluate
    print("\nRunning evaluation...")
    predictions, labels, probs = evaluate(model, data_loader, device)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Per-class metrics
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(
        labels, predictions,
        target_names=config.CLASSES,
        digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print("-" * 60)
    print(f"              Predicted")
    print(f"              {config.CLASSES[0]:<10} {config.CLASSES[1]:<10}")
    print(f"Actual {config.CLASSES[0]:<8} {cm[0][0]:<10} {cm[0][1]:<10}")
    print(f"       {config.CLASSES[1]:<8} {cm[1][0]:<10} {cm[1][1]:<10}")

    # Save confusion matrix plot
    cm_path = os.path.join(os.path.dirname(args.model), 'confusion_matrix.png')
    plot_confusion_matrix(cm, config.CLASSES, cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
