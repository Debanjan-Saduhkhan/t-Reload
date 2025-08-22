from __future__ import annotations
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .config import ModelConfig

logger = logging.getLogger(__name__)

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for t-RELOAD recommendation system.
    
    This implements the Q-function approximation for the reinforcement
    learning component of the t-RELOAD framework.
    """
    
    def __init__(self, config: ModelConfig, num_actions: int, state_dim: int):
        """
        Initialize DQN network.
        
        Args:
            config: Model configuration
            num_actions: Number of possible actions (recommendations)
            state_dim: Dimension of state representation
        """
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.state_dim = state_dim
        
        # State processing layers
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
        """
        Forward pass to compute Q-values.
        
        Args:
            state: State representation tensor [batch_size, state_dim]
            
        Returns:
            Q-values for all actions [batch_size, num_actions]
        """
        # Encode state
        state_features = self.state_encoder(state)
        
        # Predict Q-values
        q_values = self.q_layers(state_features)
        
        return q_values


class DQNAgent:
    """
    DQN Agent for t-RELOAD recommendation system.
    
    This agent implements the reinforcement learning component that
    learns to make optimal recommendations based on user engagement.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_actions: int,
        state_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100
    ):
        """
        Initialize DQN agent.
        
        Args:
            config: Model configuration
            num_actions: Number of possible actions
            state_dim: Dimension of state representation
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Size of replay buffer
            batch_size: Training batch size
            target_update: Frequency of target network updates
        """
        self.config = config
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.q_network = DQNNetwork(config, num_actions, state_dim)
        self.target_network = DQNNetwork(config, num_actions, state_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training counters
        self.step_count = 0
        
        logger.info(f"Initialized DQN agent with {num_actions} actions")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.num_actions)
        else:
            # Exploitation: best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def train(self) -> Optional[float]:
        """
        Train the DQN agent.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        experiences = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.numpy().squeeze()
    
    def save_model(self, path: str) -> None:
        """Save the DQN model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'config': self.config.to_dict()
        }, path)
        logger.info(f"DQN model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load the DQN model."""
        checkpoint = torch.load(path, map_location='cpu')
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.step_count = checkpoint.get('step_count', 0)
        logger.info(f"DQN model loaded from {path}")


class TReloadEnvironment:
    """
    Environment for t-RELOAD recommendation system.
    
    This simulates the recommendation environment where the agent
    makes recommendations and receives feedback based on user engagement.
    """
    
    def __init__(self, num_items: int, num_users: int = 1000):
        """
        Initialize recommendation environment.
        
        Args:
            num_items: Number of items to recommend
            num_users: Number of users in the environment
        """
        self.num_items = num_items
        self.num_users = num_users
        
        # Simulate user preferences (random for demo)
        self.user_preferences = np.random.rand(num_users, num_items)
        
        # Current user and state
        self.current_user = None
        self.current_state = None
        self.user_history = []
        
        logger.info(f"Initialized recommendation environment with {num_items} items and {num_users} users")
    
    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        """
        Reset environment for a new user session.
        
        Args:
            user_id: User ID (random if None)
            
        Returns:
            Initial state representation
        """
        if user_id is None:
            user_id = np.random.randint(0, self.num_users)
        
        self.current_user = user_id
        self.user_history = []
        
        # Create initial state (user preferences + empty history)
        state = np.concatenate([
            self.user_preferences[user_id],
            np.zeros(self.num_items)  # Empty history
        ])
        
        self.current_state = state
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action and get feedback.
        
        Args:
            action: Item index to recommend
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.current_user is None:
            raise ValueError("Environment not reset")
        
        # Simulate user engagement based on preferences
        user_pref = self.user_preferences[self.current_user, action]
        
        # Reward based on user preference (higher preference = higher reward)
        reward = user_pref * 10.0  # Scale reward
        
        # Add some randomness to simulate real-world behavior
        reward += np.random.normal(0, 0.1)
        
        # Update user history
        self.user_history.append(action)
        
        # Create next state
        history_vector = np.zeros(self.num_items)
        for item in self.user_history:
            history_vector[item] = 1.0
        
        next_state = np.concatenate([
            self.user_preferences[self.current_user],
            history_vector
        ])
        
        # Check if session should end (random termination)
        done = np.random.random() < 0.1  # 10% chance to end session
        
        info = {
            'user_id': self.current_user,
            'action': action,
            'user_preference': user_pref,
            'history_length': len(self.user_history)
        }
        
        self.current_state = next_state
        return next_state, reward, done, info
    
    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.num_items * 2  # preferences + history


def create_dqn_agent(config: ModelConfig, num_items: int) -> Tuple[DQNAgent, TReloadEnvironment]:
    """
    Create DQN agent and environment for t-RELOAD.
    
    Args:
        config: Model configuration
        num_items: Number of items to recommend
        
    Returns:
        Tuple of (DQN agent, environment)
    """
    # Create environment
    env = TReloadEnvironment(num_items=num_items)
    state_dim = env.get_state_dim()
    
    # Create DQN agent
    agent = DQNAgent(
        config=config,
        num_actions=num_items,
        state_dim=state_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    return agent, env
