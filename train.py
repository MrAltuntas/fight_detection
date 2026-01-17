#!/usr/bin/env python3
"""
Training script for violence detection model.

Usage:
    python train.py
    python train.py --epochs 10 --batch_size 4 --lr 0.0001
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from models import ViolenceDetector
from data.dataset import get_dataloaders
from utils.helpers import (
    get_device,
    save_checkpoint,
    load_checkpoint,
    calculate_accuracy,
    plot_training_curves
)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True  # Optimize for fixed input size (pre-processed tensors)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple:
    """
    Train model for one epoch.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for frames, labels in progress_bar:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Validate model on validation set.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)

        for frames, labels in progress_bar:
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train violence detection model')

    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(config.SEED)
    print(f"Random seed set to: {config.SEED}")

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = os.path.dirname(config.MODEL_SAVE_PATH)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create DataLoaders
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Initialize model
    model = ViolenceDetector(
        num_frames=config.NUM_FRAMES,
        num_classes=config.NUM_CLASSES,
        lstm_hidden=config.LSTM_HIDDEN
    )
    model = model.to(device)

    # Debug: Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Loss function and optimizer (only trainable parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        #verbose=True
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        metadata = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch = metadata['epoch'] + 1
        best_val_acc = metadata['val_accuracy']
        print(f"Resumed from epoch {metadata['epoch']}, val_acc: {best_val_acc:.4f}")

    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Early stopping counter
    patience_counter = 0

    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print("=" * 60)

    for epoch in range(start_epoch, config.EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch + 1}/{config.EPOCHS}] | LR: {current_lr:.6f}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch summary
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Update learning rate scheduler
        scheduler.step(val_acc)

        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                train_loss, val_loss, val_acc,
                config.MODEL_SAVE_PATH
            )
            print(f"  -> Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        # Early stopping check
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")

    # Plot training curves
    if len(train_losses) > 0:
        curves_path = os.path.join(checkpoint_dir, 'training_curves.png')
        plot_training_curves(
            train_losses, val_losses,
            train_accs, val_accs,
            curves_path
        )
        print(f"Training curves saved to: {curves_path}")


if __name__ == "__main__":
    main()
