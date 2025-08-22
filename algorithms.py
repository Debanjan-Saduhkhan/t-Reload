"""
Implementation of the exact algorithms from the t-RELOAD paper.

This module implements:
- Algorithm 1: Offline training (using relevancy)
- Algorithm 2: Online training (using turbo-reward for engagement)
"""

import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, namedtuple

from .config import ModelConfig

logger = logging.getLogger(__name__)

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for t-RELOAD as described in the paper.
    
    The paper mentions using Double-DQN with Noise Clipping for better performance.
    """
    
    def __init__(self, config: ModelConfig, num_actions: int, state_dim: int):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.state_dim = state_dim
        
        # State processing layers (based on paper's architecture)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Q-value prediction layers
        self.q_layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, num_actions)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized DQN with {num_actions} actions and {state_dim} state dimensions")
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values."""
        state_features = self.state_encoder(state)
        q_values = self.q_layers(state_features)
        return q_values


class TReloadOfflineTrainer:
    """
    Implementation of Algorithm 1: Offline Training from the t-RELOAD paper.
    
    This implements the offline training process using relevancy-based rewards
    as described in the paper.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_actions: int,
        state_dim: int,
        learning_rate: float = 0.00005,  # From paper Table 2
        gamma: float = 0.7,              # From paper Table 2
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,       # From paper Table 2
        epsilon_decay: float = 0.05,     # From paper Table 2
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 10          # From paper Table 2
    ):
        """
        Initialize offline trainer following Algorithm 1.
        
        Args:
            config: Model configuration
            num_actions: Number of possible actions (0, 1, 2 for agent-1)
            state_dim: Dimension of state representation (124 from paper)
            learning_rate: Learning rate (0.00005 from paper)
            gamma: Discount factor (0.7 from paper)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate (0.01 from paper)
            epsilon_decay: Exploration decay rate (0.05 from paper)
            buffer_size: Size of replay buffer
            batch_size: Training batch size
            target_update: Frequency of target network updates (10 from paper)
        """
        self.config = config
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks (policy and target)
        self.policy_network = DQNNetwork(config, num_actions, state_dim)
        self.target_network = DQNNetwork(config, num_actions, state_dim)
        
        # Copy weights from policy to target network (step 1 of Algorithm 1)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training counters
        self.step_count = 0
        self.iteration_count = 0
        
        logger.info("Initialized t-RELOAD Offline Trainer following Algorithm 1")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using exploration/exploitation (step 5 of Algorithm 1).
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action index (0, 1, or 2 for agent-1)
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploitation: best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_network(state_tensor)
                return q_values.argmax().item()
    
    def estimate_reward(self, state: np.ndarray, action_rec: int, action_actual: int, 
                       lambda_val: float = 0.5, beta: float = 0.1) -> float:
        """
        Estimate reward following Equation 2 from the paper.
        
        r(s, a_rec, a_actual) = λ × a_actual if a_actual == a_rec
                               = -β|a_actual - a_rec| × a_actual otherwise
        
        Args:
            state: Current state
            action_rec: Recommended action
            action_actual: Actual action taken
            lambda_val: Lambda parameter (0.5 from paper)
            beta: Beta parameter (0.1 from paper)
            
        Returns:
            Estimated reward
        """
        if action_rec == action_actual:
            # Positive reward if recommendation matches actual
            return lambda_val * action_actual
        else:
            # Negative reward based on difference
            return -beta * abs(action_actual - action_rec) * action_actual
    
    def compute_target(self, reward: float, next_state: np.ndarray) -> float:
        """
        Compute target Q-value following step 8 of Algorithm 1.
        
        target_y = [r(s, a_rec, a_actual) + γQ*(s', argmax_a' Q*(s', a'; θ); θ') + ε]
        
        Args:
            reward: Estimated reward
            next_state: Next state
            
        Returns:
            Target Q-value
        """
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Q*(s', argmax_a' Q*(s', a'; θ); θ')
            q_values_next = self.policy_network(next_state_tensor)
            best_action_next = q_values_next.argmax()
            
            target_q_values = self.target_network(next_state_tensor)
            max_q_next = target_q_values[0, best_action_next].item()
            
            # Add small noise ε for stability
            epsilon_noise = np.random.normal(0, 0.01)
            
            target_y = reward + self.gamma * max_q_next + epsilon_noise
            return target_y
    
    def train_policy_network(self, states: torch.Tensor, actions: torch.Tensor, 
                           targets: torch.Tensor) -> float:
        """
        Train policy network using Equation 4 from the paper (step 10 of Algorithm 1).
        
        Args:
            states: Batch of states
            actions: Batch of actions
            targets: Batch of target Q-values
            
        Returns:
            Loss value
        """
        # Get current Q-values
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss (mean squared error as mentioned in paper)
        loss = F.mse_loss(current_q_values.squeeze(), targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_offline(self, batch_data: List[Tuple]) -> float:
        """
        Execute Algorithm 1: Offline Training.
        
        Args:
            batch_data: List of (state, action_actual, next_state, reward) tuples
            
        Returns:
            Average loss
        """
        total_loss = 0.0
        num_samples = len(batch_data)
        
        # Step 3: for batch = 1, 2, ..., N do
        for i, (state, action_actual, next_state, reward) in enumerate(batch_data):
            # Step 4: for each new training sample (s, a_actual, s', r) do
            
            # Step 5: Select an action either by exploration or exploitation
            action_rec = self.select_action(state, training=True)
            
            # Step 6: Set a_rec = action from explore/exploitation
            # (action_rec is already set)
            
            # Step 7: Estimate reward r(s, a_rec, a_actual) following Eq. 2
            estimated_reward = self.estimate_reward(state, action_rec, action_actual)
            
            # Step 8: Estimate target_y = [r(s, a_rec, a_actual) + γQ*(s', argmax_a' Q*(s', a'; θ); θ') + ε]
            target_y = self.compute_target(estimated_reward, next_state)
            
            # Store experience for replay
            experience = Experience(state, action_rec, estimated_reward, next_state, False)
            self.memory.push(experience)
            
            # Step 10: Train policy network using Eq. 4
            if len(self.memory) >= self.batch_size:
                # Sample batch from replay buffer
                experiences = self.memory.sample(self.batch_size)
                
                # Convert to tensors
                batch_states = torch.FloatTensor([e.state for e in experiences])
                batch_actions = torch.LongTensor([e.action for e in experiences])
                batch_targets = torch.FloatTensor([e.reward for e in experiences])
                
                # Train policy network
                loss = self.train_policy_network(batch_states, batch_actions, batch_targets)
                total_loss += loss
        
        # Step 11: After some iteration, copy weights from policy network to target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            logger.info(f"Target network updated at step {self.step_count}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
        
        avg_loss = total_loss / max(num_samples, 1)
        return avg_loss


class TReloadOnlineTrainer:
    """
    Implementation of Algorithm 2: Online Training from the t-RELOAD paper.
    
    This implements the online training process using turbo-reward for engagement
    as described in the paper.
    """
    
    def __init__(
        self,
        offline_trainer: TReloadOfflineTrainer,
        learning_rate: float = 0.00005  # From paper Table 2
    ):
        """
        Initialize online trainer following Algorithm 2.
        
        Args:
            offline_trainer: Pre-trained offline trainer
            learning_rate: Learning rate for online updates
        """
        self.offline_trainer = offline_trainer
        self.learning_rate = learning_rate
        
        # Step 2: Weights are copied using Algo 1 for both policy and target network
        # (This is already done in the offline trainer)
        
        logger.info("Initialized t-RELOAD Online Trainer following Algorithm 2")
    
    def compute_turbo_reward(self, engagement_metrics: Dict[str, float]) -> float:
        """
        Compute turbo-reward based on platform objectives (engagement).
        
        The paper mentions that if number-of-games increases and inter-session-duration
        decreases, the overall engagement increases.
        
        Args:
            engagement_metrics: Dictionary containing engagement metrics
            
        Returns:
            Turbo-reward value
        """
        num_games = engagement_metrics.get('num_games', 0)
        inter_session_duration = engagement_metrics.get('inter_session_duration', 1.0)
        
        # Higher reward for more games and lower inter-session duration
        games_factor = min(num_games / 10.0, 1.0)  # Normalize to [0, 1]
        duration_factor = max(0, 1.0 - inter_session_duration / 60.0)  # Normalize to [0, 1]
        
        # Combine factors for overall engagement reward
        turbo_reward = (games_factor + duration_factor) / 2.0
        
        return turbo_reward
    
    def train_online(self, batch_data: List[Tuple], engagement_metrics: List[Dict[str, float]]) -> float:
        """
        Execute Algorithm 2: Online Training.
        
        Args:
            batch_data: List of (state, action_actual, next_state, reward) tuples
            engagement_metrics: List of engagement metrics for each sample
            
        Returns:
            Average loss
        """
        total_loss = 0.0
        num_samples = len(batch_data)
        
        # Step 3: for each batch = 1, 2, ..., N do
        for i, ((state, action_actual, next_state, reward), metrics) in enumerate(zip(batch_data, engagement_metrics)):
            # Step 4: Generate actions either by exploration/exploitation
            action_rec = self.offline_trainer.select_action(state, training=True)
            
            # Step 5: Estimate platform-centric reward r (e.g., engagement)
            turbo_reward = self.compute_turbo_reward(metrics)
            
            # Step 6: Update the weights of policy network using r and ε
            # Use the offline trainer's networks but with turbo-reward
            estimated_reward = self.offline_trainer.estimate_reward(state, action_rec, action_actual)
            
            # Combine relevancy reward with turbo-reward
            combined_reward = estimated_reward + turbo_reward
            
            # Compute target and train
            target_y = self.offline_trainer.compute_target(combined_reward, next_state)
            
            # Store experience
            experience = Experience(state, action_rec, combined_reward, next_state, False)
            self.offline_trainer.memory.push(experience)
            
            # Train if enough samples
            if len(self.offline_trainer.memory) >= self.offline_trainer.batch_size:
                experiences = self.offline_trainer.memory.sample(self.offline_trainer.batch_size)
                
                batch_states = torch.FloatTensor([e.state for e in experiences])
                batch_actions = torch.LongTensor([e.action for e in experiences])
                batch_targets = torch.FloatTensor([e.reward for e in experiences])
                
                loss = self.offline_trainer.train_policy_network(batch_states, batch_actions, batch_targets)
                total_loss += loss
        
        # Step 7: After some iteration, copy weights from policy network to target network
        self.offline_trainer.step_count += 1
        if self.offline_trainer.step_count % self.offline_trainer.target_update == 0:
            self.offline_trainer.target_network.load_state_dict(self.offline_trainer.policy_network.state_dict())
            logger.info(f"Target network updated during online training at step {self.offline_trainer.step_count}")
        
        avg_loss = total_loss / max(num_samples, 1)
        return avg_loss


def create_treload_trainers(
    config: ModelConfig,
    num_actions: int = 3,  # 0, 1, 2 for agent-1 as per paper
    state_dim: int = 124   # From paper Table 2
) -> Tuple[TReloadOfflineTrainer, TReloadOnlineTrainer]:
    """
    Create t-RELOAD offline and online trainers.
    
    Args:
        config: Model configuration
        num_actions: Number of actions (3 for agent-1)
        state_dim: State dimension (124 from paper)
        
    Returns:
        Tuple of (offline_trainer, online_trainer)
    """
    # Create offline trainer (Algorithm 1)
    offline_trainer = TReloadOfflineTrainer(
        config=config,
        num_actions=num_actions,
        state_dim=state_dim
    )
    
    # Create online trainer (Algorithm 2)
    online_trainer = TReloadOnlineTrainer(offline_trainer)
    
    return offline_trainer, online_trainer
