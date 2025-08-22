#!/usr/bin/env python3
"""
Training script for the DQN reinforcement learning component of t-RELOAD.

This script implements the reinforcement learning training pipeline
for the recommendation system using Deep Q-Networks.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from paper_impl.config import ExperimentConfig, get_default_config
from paper_impl.dqn import create_dqn_agent
from paper_impl.utils import set_seed, ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_dqn_episode(
    agent,
    env,
    max_steps: int = 100
) -> Dict[str, float]:
    """
    Train DQN agent for one episode.
    
    Args:
        agent: DQN agent
        env: Environment
        max_steps: Maximum steps per episode
        
    Returns:
        Episode statistics
    """
    state = env.reset()
    total_reward = 0.0
    steps = 0
    
    for step in range(max_steps):
        # Select action
        action = agent.select_action(state, training=True)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, done)
        
        # Train agent
        loss = agent.train()
        
        # Update state and reward
        state = next_state
        total_reward += reward
        steps += 1
        
        # Log progress
        if step % 10 == 0:
            loss_str = f"{loss:.3f}" if loss is not None else "0.0"
            logger.info(f"Step {step}: Action={action}, Reward={reward:.3f}, Loss={loss_str}")
        
        if done:
            break
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'epsilon': agent.epsilon
    }


def evaluate_dqn_agent(
    agent,
    env,
    num_episodes: int = 10,
    max_steps: int = 50
) -> Dict[str, float]:
    """
    Evaluate DQN agent.
    
    Args:
        agent: DQN agent
        env: Environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        
    Returns:
        Evaluation statistics
    """
    total_rewards = []
    total_steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            # Select action (no exploration during evaluation)
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        logger.info(f"Evaluation Episode {episode}: Reward={episode_reward:.3f}, Steps={steps}")
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_steps': np.mean(total_steps),
        'std_steps': np.std(total_steps)
    }


def plot_training_curves(
    rewards: List[float],
    losses: List[float],
    epsilons: List[float],
    save_path: str
) -> None:
    """Plot training curves."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    ax2.plot(losses)
    ax2.set_title('Training Losses')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Plot epsilon
    ax3.plot(epsilons)
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {save_path}")


def main(args: argparse.Namespace) -> None:
    """Main training function for DQN."""
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
    
    # Create DQN agent and environment
    logger.info("Creating DQN agent and environment...")
    agent, env = create_dqn_agent(config.model, num_items=args.num_items)
    
    # Training statistics
    episode_rewards = []
    episode_losses = []
    episode_epsilons = []
    
    # Training loop
    logger.info(f"Starting DQN training for {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        # Train one episode
        episode_stats = train_dqn_episode(agent, env, max_steps=args.max_steps)
        
        # Store statistics
        episode_rewards.append(episode_stats['total_reward'])
        episode_epsilons.append(episode_stats['epsilon'])
        
        # Log progress
        logger.info(f"Episode {episode + 1}/{args.num_episodes}: "
                   f"Reward={episode_stats['total_reward']:.3f}, "
                   f"Steps={episode_stats['steps']}, "
                   f"Epsilon={episode_stats['epsilon']:.3f}")
        
        # Evaluate periodically
        if (episode + 1) % args.eval_interval == 0:
            logger.info("Evaluating agent...")
            eval_stats = evaluate_dqn_agent(agent, env, num_episodes=5, max_steps=args.max_steps)
            logger.info(f"Evaluation: Mean Reward={eval_stats['mean_reward']:.3f} ± {eval_stats['std_reward']:.3f}")
        
        # Save model periodically
        if (episode + 1) % args.save_interval == 0:
            model_path = output_dir / f"dqn_episode_{episode + 1}.pt"
            agent.save_model(str(model_path))
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_eval = evaluate_dqn_agent(agent, env, num_episodes=20, max_steps=args.max_steps)
    logger.info(f"Final Evaluation: Mean Reward={final_eval['mean_reward']:.3f} ± {final_eval['std_reward']:.3f}")
    
    # Save final model
    final_model_path = output_dir / "dqn_final_model.pt"
    agent.save_model(str(final_model_path))
    
    # Plot training curves
    plot_path = output_dir / "dqn_training_curves.png"
    plot_training_curves(episode_rewards, episode_losses, episode_epsilons, str(plot_path))
    
    # Save training statistics
    import json
    stats = {
        'episode_rewards': episode_rewards,
        'episode_epsilons': episode_epsilons,
        'final_evaluation': final_eval,
        'config': config.to_dict()
    }
    
    stats_path = output_dir / "dqn_training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info("DQN training completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN for t-RELOAD")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="dqn_experiment",
        help="Name of the experiment"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dqn",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--num-items",
        type=int,
        default=50,
        help="Number of items to recommend"
    )
    
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of training episodes"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode"
    )
    
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluation interval (episodes)"
    )
    
    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
        help="Model save interval (episodes)"
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
