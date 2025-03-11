import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Import our business environment
from business_simulator.business_environment import BusinessEnvironment
from trainer import business_simulator
from business_simulator.model_store import ModelStore


class DQNetwork(nn.Module):
    """Deep Q-Network model implemented in PyTorch."""
    
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        # Wider network for better feature extraction
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.batch_norm1 = nn.BatchNorm1d(128)  # Add batch normalization
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        # Enhanced forward pass with regularization
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)


class DQNAgent:
    """
    Deep Q-Network (DQN) agent for learning optimal business decisions.
    
    This implementation uses:
    - Experience replay to stabilize learning
    - Target network to reduce oscillations
    - Double DQN to reduce overestimation bias
    """
    
    def __init__(
        self,
        env,
        memory_size=10000,
        batch_size=64,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        target_update_freq=10,
        checkpoint_dir="models"
    ):
        """
        Initialize the DQN agent with hyperparameters.
        
        Args:
            env: The business environment
            memory_size: Size of replay memory buffer
            batch_size: Number of samples per training batch
            gamma: Discount factor for future rewards
            epsilon: Exploration rate (probability of random action)
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays over time
            learning_rate: Learning rate for optimizer
            target_update_freq: Frequency to update target network
            checkpoint_dir: Directory to save model checkpoints
        """
        self.env = env
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Setup action mappings first since we need them to determine action size
        self.setup_action_mappings()
        
        # Calculate action size based on the number of combinations
        self.action_size = len(self.price_levels) * len(self.production_levels) * len(self.marketing_levels)
        
        # Input shape is the observation space shape
        self.state_size = env.observation_space.shape[0]
        
        # Set device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create main and target networks
        self.model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.update_target_network()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Training counters
        self.train_step = 0
        
        # Metrics
        self.metrics = {
            'losses': [],
            'rewards': [],
            'epsilons': []
        }
    
    def setup_action_mappings(self):
        """Define mappings from discrete actions to continuous values."""
        
        # Define price levels (as percentages of cost)
        self.price_levels = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]  # Increased granularity
        
        # Define production quantity levels (as percentages of max production)
        self.production_levels = [0.0, 0.25, 0.5, 0.75, 1.0]  # Increased granularity
        
        # Define marketing budget levels (as percentages of max budget)
        self.marketing_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Increased granularity
        
        # Create action mapping
        self.action_mapping = {}
        idx = 0
        for p in range(len(self.price_levels)):
            for q in range(len(self.production_levels)):
                for m in range(len(self.marketing_levels)):
                    self.action_mapping[idx] = (p, q, m)
                    idx += 1
    
    def update_target_network(self):
        """Update target network weights with current model weights."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_discrete_action(self, state_tensor):
        """Choose a discrete action index based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Set model to evaluation mode for inference
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state_tensor)
            return torch.argmax(act_values).item()
        # Set model back to training mode
        self.model.train()
    
    def map_to_continuous_action(self, discrete_action):
        """Map discrete action index to continuous action values."""
        p_idx, q_idx, m_idx = self.action_mapping[discrete_action]
        
        # Map price (as percentage of cost)
        price_factor = self.price_levels[p_idx]
        price = self.env.current_cost * price_factor
        
        # Map production quantity
        prod_factor = self.production_levels[q_idx]
        production_qty = self.env.config['max_production'] * prod_factor
        
        # Map marketing budget
        mktg_factor = self.marketing_levels[m_idx]
        marketing_budget = self.env.config['max_marketing_budget'] * mktg_factor
        
        return np.array([price, production_qty, marketing_budget])
    
    def act(self, state):
        """
        Choose an action based on the current state.
        Maps from state -> discrete action -> continuous action.
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Choose a discrete action
        discrete_action = self.choose_discrete_action(state_tensor)
        
        # Map to continuous action
        continuous_action = self.map_to_continuous_action(discrete_action)
        
        # Store the discrete action for learning
        self.last_discrete_action = discrete_action
        
        return continuous_action
    
    def replay(self):
        """Train the model with experiences from replay memory."""
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples
        
        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract batch data
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Double DQN implementation
        # 1. Get actions from main network
        with torch.no_grad():
            max_actions = torch.argmax(self.model(next_states_tensor), dim=1)
        
        # 2. Get Q values for those actions from target network
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states_tensor).gather(1, max_actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q values
        target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_values
        
        # Compute loss and optimize
        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network if needed
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Store metrics
        loss_value = loss.item()
        self.metrics['losses'].append(loss_value)
        self.metrics['epsilons'].append(self.epsilon)
        
        return loss_value
    
    def learn(self, state, action, reward, next_state, done):
        """Learn from an experience tuple."""
        # Remember the discrete action
        self.remember(state, self.last_discrete_action, reward, next_state, done)
        
        # Store reward for metrics
        if len(self.metrics['rewards']) == 0:
            self.metrics['rewards'].append(reward)
        else:
            self.metrics['rewards'].append(self.metrics['rewards'][-1] + reward)
        
        # Train on a batch of experiences
        return self.replay()
    
    def save(self, filename=None):
        """Save the model to disk using ModelStore for versioning and validation."""
        if filename is None:
            filename = f"dqn_agent_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize ModelStore
        model_store = ModelStore(self.checkpoint_dir)
        model_store.register_model_class('DQNetwork', DQNetwork)
        
        # Save model with metadata
        filepath = model_store.save(
            model=self.model,
            model_name=filename,
            optimizer=self.optimizer,
            hyperparameters={
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'learning_rate': self.learning_rate,
                'state_size': self.state_size,
                'action_size': self.action_size
            },
            metrics={
                'losses': [float(loss) for loss in self.metrics['losses']],
                'rewards': [float(reward) for reward in self.metrics['rewards']],
                'epsilons': [float(epsilon) for epsilon in self.metrics['epsilons']]
            }
        )
        
        return filepath
    
    def load(self, filepath):
        """Load a saved model from disk with architecture validation."""
        # Initialize ModelStore
        model_store = ModelStore(self.checkpoint_dir)
        model_store.register_model_class('DQNetwork', DQNetwork)
        
        try:
            # Load model with architecture validation
            self.model, metadata = model_store.load(
                model_instance=self.model,
                model_name=os.path.basename(filepath),
                optimizer=self.optimizer
            )
            
            # Update target model
            self.target_model.load_state_dict(self.model.state_dict())
            
            # Load hyperparameters from metadata
            if 'hyperparameters' in metadata:
                hyperparams = metadata['hyperparameters']
                self.memory_size = hyperparams.get('memory_size', self.memory_size)
                self.batch_size = hyperparams.get('batch_size', self.batch_size)
                self.gamma = hyperparams.get('gamma', self.gamma)
                self.epsilon = hyperparams.get('epsilon', self.epsilon)
                self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
                self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
                self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
            
            # Load metrics if available
            if 'metrics' in metadata:
                self.metrics = metadata['metrics']
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def plot_metrics(self):
        """Plot learning metrics."""
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(self.metrics['losses'])
        plt.title('Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        
        # Plot rewards
        plt.subplot(1, 3, 2)
        plt.plot(self.metrics['rewards'])
        plt.title('Cumulative Reward')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        
        # Plot epsilon
        plt.subplot(1, 3, 3)
        plt.plot(self.metrics['epsilons'])
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Training Steps')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()