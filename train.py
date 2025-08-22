#!/usr/bin/env python3
"""
Training script for the t-reload model implementation.

This script implements the training pipeline described in the research paper,
including the t-reload functionality for parameter reloading during training.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from paper_impl.config import ExperimentConfig, get_default_config
from paper_impl.model import TReloadModel, TReloadTrainer
from paper_impl.data import TReloadDataLoader, create_sample_data
from paper_impl.utils import set_seed, ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(
    trainer: TReloadTrainer,
    train_loader: DataLoader,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Training step
        step_results = trainer.train_step(batch)
        loss = step_results['loss']
        total_loss += loss
        
        # Log progress
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")
        
        # TensorBoard logging
        if writer:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/Train', loss, global_step)
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    return {'train_loss': avg_loss}


def evaluate(
    trainer: TReloadTrainer,
    val_loader: DataLoader,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Evaluate the model."""
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(val_loader)
    
    trainer.model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            step_results = trainer.evaluate_step(batch)
            total_loss += step_results['loss']
            
            if 'accuracy' in step_results:
                total_accuracy += step_results['accuracy']
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    
    logger.info(f"Validation - Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    # TensorBoard logging
    if writer:
        writer.add_scalar('Loss/Val', avg_loss, epoch)
        writer.add_scalar('Accuracy/Val', avg_accuracy, epoch)
    
    return {'val_loss': avg_loss, 'val_accuracy': avg_accuracy}


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    
    # Create output directories
    output_dir = Path(args.output_dir)
    ensure_dir(str(output_dir))
    
    # Setup TensorBoard
    log_dir = output_dir / "logs"
    ensure_dir(str(log_dir))
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Create sample data if it doesn't exist
    data_dir = Path(config.data.data_dir)
    if not data_dir.exists() or args.create_sample_data:
        logger.info("Creating sample data...")
        create_sample_data(data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = TReloadDataLoader.create_dataloaders(
        train_path=data_dir / config.data.train_file,
        val_path=data_dir / config.data.val_file,
        test_path=data_dir / config.data.test_file,
        config=config.training.to_dict()
    )
    
    # Create model
    model = TReloadModel(config.model)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = TReloadTrainer(model, config.training.to_dict())
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = output_dir / "checkpoints"
    ensure_dir(str(checkpoint_dir))
    
    for epoch in range(config.training.num_epochs):
        # Train
        train_metrics = train_epoch(trainer, train_loader, epoch, writer)
        
        # Evaluate
        if val_loader:
            val_metrics = evaluate(trainer, val_loader, epoch, writer)
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_model_path = checkpoint_dir / "best_model.pt"
                trainer.model.save_checkpoint(str(best_model_path), trainer.optimizer, epoch)
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if epoch % config.training.save_steps == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.model.save_checkpoint(str(checkpoint_path), trainer.optimizer, epoch)
        
        # Learning rate scheduling
        trainer.scheduler.step()
        
        # Log learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    # Save final model
    final_model_path = checkpoint_dir / "final_model.pt"
    trainer.model.save_checkpoint(str(final_model_path), trainer.optimizer, config.training.num_epochs)
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Output saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train t-reload model")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name of the experiment"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample data if it doesn't exist"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
