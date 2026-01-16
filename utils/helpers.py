"""
Helper utilities for model training and evaluation.

Functions:
- save_checkpoint: Save model state to file
- load_checkpoint: Load model state from file
- calculate_accuracy: Calculate accuracy from model outputs
- plot_training_curves: Plot and save training curves
"""

import torch
import matplotlib.pyplot as plt
from typing import Optional, Dict, List


def get_device() -> torch.device:
    """
    Get the best available device for training.

    Returns:
        torch.device: CUDA if available, MPS for Apple Silicon, or CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    path: str
) -> None:
    """
    Save model checkpoint to file.

    Args:
        model: PyTorch model
        optimizer: Optimizer used for training
        epoch: Current epoch number
        train_loss: Training loss at this epoch
        val_loss: Validation loss at this epoch
        val_accuracy: Validation accuracy at this epoch
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint from file.

    Args:
        model: PyTorch model to load weights into
        path: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        device: Device to load the model to

    Returns:
        Dictionary with metadata: epoch, train_loss, val_loss, val_accuracy
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', 0.0),
        'val_loss': checkpoint.get('val_loss', 0.0),
        'val_accuracy': checkpoint.get('val_accuracy', 0.0)
    }


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate accuracy from model outputs and ground truth labels.

    Args:
        outputs: Model output logits of shape (batch, num_classes)
        labels: Ground truth labels of shape (batch,)

    Returns:
        Accuracy as a float between 0 and 1
    """
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str
) -> None:
    """
    Plot and save training/validation loss and accuracy curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
