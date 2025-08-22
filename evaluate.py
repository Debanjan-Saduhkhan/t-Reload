#!/usr/bin/env python3
"""
Evaluation script for the t-reload model implementation.

This script evaluates trained models on test data and provides
comprehensive metrics and analysis.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from paper_impl.config import ExperimentConfig
from paper_impl.model import TReloadModel
from paper_impl.data import TReloadDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: TReloadModel,
    test_loader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate the model on test data.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Get predictions
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            if 'loss' in outputs:
                all_losses.append(outputs['loss'].item())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Classification report
    class_report = classification_report(
        all_labels, 
        all_predictions, 
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Average loss
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    
    results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return results


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None
) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(conf_matrix))]
    
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str
) -> None:
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {save_path}")


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    experiment_name: str
) -> None:
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = output_path / f"{experiment_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed classification report
    report_path = output_path / f"{experiment_name}_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Loss: {results['loss']:.4f}\n\n")
        
        # Detailed report
        class_report = results['classification_report']
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):
                f.write(f"{class_name}:\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
                f.write("\n")
    
    # Plot confusion matrix
    conf_matrix = np.array(results['confusion_matrix'])
    cm_path = output_path / f"{experiment_name}_confusion_matrix.png"
    plot_confusion_matrix(conf_matrix, str(cm_path))
    
    logger.info(f"Results saved to {output_path}")


def main(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        logger.error("Configuration file is required")
        return
    
    # Setup device
    device = torch.device(args.device if args.device else config.device)
    logger.info(f"Using device: {device}")
    
    # Load test data
    test_loader, _, _ = TReloadDataLoader.create_dataloaders(
        train_path="",  # Not needed for evaluation
        test_path=Path(config.data.data_dir) / config.data.test_file,
        config=config.training.to_dict()
    )
    
    if test_loader is None:
        logger.error("Test data not found")
        return
    
    # Load model
    model = TReloadModel(config.model)
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
    else:
        logger.error("Checkpoint path is required")
        return
    
    # Move model to device
    model.to(device)
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    logger.info(f"Evaluation completed!")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Loss: {results['loss']:.4f}")
    
    # Save results
    if args.output_dir:
        save_results(results, args.output_dir, config.experiment_name)
    
    # Print detailed classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    
    class_report = results['classification_report']
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):
            print(f"\n{class_name}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate t-reload model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for evaluation"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
