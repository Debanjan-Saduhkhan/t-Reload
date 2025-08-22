#!/usr/bin/env python3
"""
Training script implementing the exact algorithms from the t-RELOAD paper.

This script follows:
- Algorithm 1: Offline training (using relevancy)
- Algorithm 2: Online training (using turbo-reward for engagement)

Based on the exact algorithm descriptions in the paper.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from paper_impl.config import ExperimentConfig, get_default_config
from paper_impl.algorithms import create_treload_trainers
from paper_impl.utils import set_seed, ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(num_samples: int = 1000, state_dim: int = 124) -> Tuple[List, List]:
    """
    Create sample data for training following the paper's specifications.
    
    Args:
        num_samples: Number of training samples
        state_dim: State dimension (124 from paper)
        
    Returns:
        Tuple of (batch_data, engagement_metrics)
    """
    batch_data = []
    engagement_metrics = []
    
    for i in range(num_samples):
        # Create random state (124 dimensions as per paper)
        state = np.random.randn(state_dim)
        
        # Random action (0, 1, or 2 for agent-1)
        action_actual = np.random.randint(0, 3)
        
        # Random next state
        next_state = np.random.randn(state_dim)
        
        # Random reward (for offline training)
        reward = np.random.normal(0, 1)
        
        batch_data.append((state, action_actual, next_state, reward))
        
        # Engagement metrics for online training
        metrics = {
            'num_games': np.random.randint(1, 20),
            'inter_session_duration': np.random.uniform(10, 120)  # minutes
        }
        engagement_metrics.append(metrics)
    
    return batch_data, engagement_metrics


def train_offline_phase(
    offline_trainer,
    num_iterations: int = 500,  # From paper Table 2
    num_epochs_per_iteration: int = 5,  # From paper Table 2
    batch_size: int = 100
) -> List[float]:
    """
    Execute Algorithm 1: Offline Training.
    
    Args:
        offline_trainer: Offline trainer instance
        num_iterations: Number of iterations (500 from paper)
        num_epochs_per_iteration: Epochs per iteration (5 from paper)
        batch_size: Batch size for training
        
    Returns:
        List of losses per iteration
    """
    logger.info(f"Starting Algorithm 1: Offline Training")
    logger.info(f"Number of iterations: {num_iterations}")
    logger.info(f"Epochs per iteration: {num_epochs_per_iteration}")
    
    iteration_losses = []
    
    for iteration in range(num_iterations):
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        
        iteration_loss = 0.0
        
        # Train for multiple epochs per iteration
        for epoch in range(num_epochs_per_iteration):
            # Create sample data for this epoch
            batch_data, _ = create_sample_data(batch_size)
            
            # Execute Algorithm 1
            loss = offline_trainer.train_offline(batch_data)
            iteration_loss += loss
            
            logger.info(f"  Epoch {epoch + 1}: Loss = {loss:.6f}")
        
        avg_iteration_loss = iteration_loss / num_epochs_per_iteration
        iteration_losses.append(avg_iteration_loss)
        
        logger.info(f"Iteration {iteration + 1} completed. Average loss: {avg_iteration_loss:.6f}")
        logger.info(f"Current epsilon: {offline_trainer.epsilon:.4f}")
        
        # Save model periodically
        if (iteration + 1) % 50 == 0:
            model_path = f"models/offline_iteration_{iteration + 1}.pt"
            torch.save({
                'iteration': iteration + 1,
                'policy_network_state_dict': offline_trainer.policy_network.state_dict(),
                'target_network_state_dict': offline_trainer.target_network.state_dict(),
                'optimizer_state_dict': offline_trainer.optimizer.state_dict(),
                'epsilon': offline_trainer.epsilon,
                'step_count': offline_trainer.step_count
            }, model_path)
            logger.info(f"Model saved to {model_path}")
    
    return iteration_losses


def train_online_phase(
    online_trainer,
    num_iterations: int = 100,  # Online training iterations
    batch_size: int = 100
) -> List[float]:
    """
    Execute Algorithm 2: Online Training.
    
    Args:
        online_trainer: Online trainer instance
        num_iterations: Number of online training iterations
        batch_size: Batch size for training
        
    Returns:
        List of losses per iteration
    """
    logger.info(f"Starting Algorithm 2: Online Training")
    logger.info(f"Number of iterations: {num_iterations}")
    
    iteration_losses = []
    
    for iteration in range(num_iterations):
        logger.info(f"Online Iteration {iteration + 1}/{num_iterations}")
        
        # Create sample data with engagement metrics
        batch_data, engagement_metrics = create_sample_data(batch_size)
        
        # Execute Algorithm 2
        loss = online_trainer.train_online(batch_data, engagement_metrics)
        iteration_losses.append(loss)
        
        logger.info(f"Online Iteration {iteration + 1} completed. Loss: {loss:.6f}")
        
        # Save model periodically
        if (iteration + 1) % 25 == 0:
            model_path = f"models/online_iteration_{iteration + 1}.pt"
            torch.save({
                'iteration': iteration + 1,
                'policy_network_state_dict': online_trainer.offline_trainer.policy_network.state_dict(),
                'target_network_state_dict': online_trainer.offline_trainer.target_network.state_dict(),
                'optimizer_state_dict': online_trainer.offline_trainer.optimizer.state_dict(),
                'epsilon': online_trainer.offline_trainer.epsilon,
                'step_count': online_trainer.offline_trainer.step_count
            }, model_path)
            logger.info(f"Online model saved to {model_path}")
    
    return iteration_losses


def plot_training_results(
    offline_losses: List[float],
    online_losses: List[float],
    save_path: str
) -> None:
    """Plot training results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot offline training losses
    ax1.plot(offline_losses, 'b-', linewidth=2)
    ax1.set_title('Algorithm 1: Offline Training Losses')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot online training losses
    ax2.plot(online_losses, 'r-', linewidth=2)
    ax2.set_title('Algorithm 2: Online Training Losses')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training plots saved to {save_path}")


def main(args: argparse.Namespace) -> None:
    """Main training function following the t-RELOAD algorithms."""
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
    
    # Create output directories
    output_dir = Path(args.output_dir)
    ensure_dir(str(output_dir))
    ensure_dir("models")
    
    # Create t-RELOAD trainers following the paper's algorithms
    logger.info("Creating t-RELOAD trainers following Algorithm 1 and 2...")
    offline_trainer, online_trainer = create_treload_trainers(
        config=config.model,
        num_actions=3,    # 0, 1, 2 for agent-1 as per paper
        state_dim=124     # From paper Table 2
    )
    
    # Phase 1: Execute Algorithm 1 - Offline Training
    logger.info("=" * 60)
    logger.info("PHASE 1: ALGORITHM 1 - OFFLINE TRAINING")
    logger.info("=" * 60)
    
    offline_losses = train_offline_phase(
        offline_trainer,
        num_iterations=args.offline_iterations,
        num_epochs_per_iteration=args.epochs_per_iteration,
        batch_size=args.batch_size
    )
    
    # Save final offline model
    final_offline_path = output_dir / "final_offline_model.pt"
    torch.save({
        'phase': 'offline',
        'policy_network_state_dict': offline_trainer.policy_network.state_dict(),
        'target_network_state_dict': offline_trainer.target_network.state_dict(),
        'optimizer_state_dict': offline_trainer.optimizer.state_dict(),
        'epsilon': offline_trainer.epsilon,
        'step_count': offline_trainer.step_count,
        'config': config.to_dict()
    }, str(final_offline_path))
    
    logger.info(f"Offline training completed. Final model saved to {final_offline_path}")
    
    # Phase 2: Execute Algorithm 2 - Online Training
    logger.info("=" * 60)
    logger.info("PHASE 2: ALGORITHM 2 - ONLINE TRAINING")
    logger.info("=" * 60)
    
    online_losses = train_online_phase(
        online_trainer,
        num_iterations=args.online_iterations,
        batch_size=args.batch_size
    )
    
    # Save final online model
    final_online_path = output_dir / "final_online_model.pt"
    torch.save({
        'phase': 'online',
        'policy_network_state_dict': online_trainer.offline_trainer.policy_network.state_dict(),
        'target_network_state_dict': online_trainer.offline_trainer.target_network.state_dict(),
        'optimizer_state_dict': online_trainer.offline_trainer.optimizer.state_dict(),
        'epsilon': online_trainer.offline_trainer.epsilon,
        'step_count': online_trainer.offline_trainer.step_count,
        'config': config.to_dict()
    }, str(final_online_path))
    
    logger.info(f"Online training completed. Final model saved to {final_online_path}")
    
    # Plot training results
    plot_path = output_dir / "treload_training_results.png"
    plot_training_results(offline_losses, online_losses, str(plot_path))
    
    # Save training statistics
    import json
    stats = {
        'offline_losses': offline_losses,
        'online_losses': online_losses,
        'final_offline_epsilon': offline_trainer.epsilon,
        'final_online_epsilon': online_trainer.offline_trainer.epsilon,
        'config': config.to_dict()
    }
    
    stats_path = output_dir / "treload_training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info("T-RELOAD TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Final offline model: {final_offline_path}")
    logger.info(f"Final online model: {final_online_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train t-RELOAD following exact algorithms from paper")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="treload_algorithms_experiment",
        help="Name of the experiment"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/treload_algorithms",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--offline-iterations",
        type=int,
        default=500,  # From paper Table 2
        help="Number of offline training iterations"
    )
    
    parser.add_argument(
        "--epochs-per-iteration",
        type=int,
        default=5,  # From paper Table 2
        help="Number of epochs per iteration"
    )
    
    parser.add_argument(
        "--online-iterations",
        type=int,
        default=100,
        help="Number of online training iterations"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
