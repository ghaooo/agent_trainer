#!/usr/bin/env python3
"""
Price Analysis Demo for Business Simulator

This script analyzes the price predictions of the DQN agent during training
to determine if the agent is consistently predicting the same price (e.g., 75).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Add the parent directory to the path so we can import the business_simulator package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from business_simulator.business_environment import BusinessEnvironment
from business_simulator.reinforcement_learning_agent import DQNAgent
from business_simulator.train_and_evaluate import train_dqn_agent

# Set matplotlib to non-interactive mode to prevent blocking
plt.ioff()

# Create output directories
os.makedirs("../models/price_analysis", exist_ok=True)
os.makedirs("../logs/price_analysis", exist_ok=True)


class PriceTrackingAgent(DQNAgent):
    """
    Extension of DQNAgent that tracks price predictions during training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.price_history = []
        self.state_history = []
        self.episode_boundaries = [0]  # Mark the start of each episode
    
    def act(self, state):
        # Get action from parent class
        action = super().act(state)
        
        # Store the price (first element of action)
        self.price_history.append(action[0])
        self.state_history.append(state.copy())
        
        return action
    
    def mark_episode_end(self):
        """Mark the end of an episode for analysis purposes."""
        self.episode_boundaries.append(len(self.price_history))
    
    def get_episode_prices(self, episode):
        """Get all prices predicted during a specific episode."""
        start_idx = self.episode_boundaries[episode]
        end_idx = self.episode_boundaries[episode + 1]
        return self.price_history[start_idx:end_idx]
    
    def get_episode_states(self, episode):
        """Get all states encountered during a specific episode."""
        start_idx = self.episode_boundaries[episode]
        end_idx = self.episode_boundaries[episode + 1]
        return self.state_history[start_idx:end_idx]
    
    def analyze_price_distribution(self):
        """Analyze the distribution of predicted prices."""
        prices = np.array(self.price_history)
        
        # Basic statistics
        stats = {
            'mean': np.mean(prices),
            'median': np.median(prices),
            'min': np.min(prices),
            'max': np.max(prices),
            'std': np.std(prices)
        }
        
        # Count occurrences of each price (rounded to nearest integer)
        price_counts = defaultdict(int)
        for price in prices:
            price_counts[round(price)] += 1
        
        # Sort by frequency
        sorted_prices = sorted(price_counts.items(), key=lambda x: x[1], reverse=True)
        
        return stats, sorted_prices
    
    def plot_price_evolution(self, save_path=None):
        """Plot how prices evolve over time and episodes."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: All prices over time
        plt.subplot(2, 2, 1)
        plt.plot(self.price_history)
        
        # Add episode boundaries
        for boundary in self.episode_boundaries[1:-1]:  # Skip first and last
            plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.3)
        
        plt.title('Price Evolution Over Time')
        plt.xlabel('Decision Step')
        plt.ylabel('Price')
        
        # Plot 2: Price distribution histogram
        plt.subplot(2, 2, 2)
        plt.hist(self.price_history, bins=30)
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        # Plot 3: Average price per episode
        episode_avg_prices = []
        episode_numbers = []
        
        for i in range(len(self.episode_boundaries) - 1):
            prices = self.get_episode_prices(i)
            if prices:  # Ensure there are prices in this episode
                episode_avg_prices.append(np.mean(prices))
                episode_numbers.append(i)
        
        plt.subplot(2, 2, 3)
        plt.plot(episode_numbers, episode_avg_prices, marker='o')
        plt.title('Average Price per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Price')
        
        # Plot 4: Price variance per episode
        episode_price_variance = []
        
        for i in range(len(self.episode_boundaries) - 1):
            prices = self.get_episode_prices(i)
            if len(prices) > 1:  # Need at least 2 points for variance
                episode_price_variance.append(np.var(prices))
            else:
                episode_price_variance.append(0)
        
        plt.subplot(2, 2, 4)
        plt.plot(episode_numbers, episode_price_variance, marker='o')
        plt.title('Price Variance per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Variance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def run_price_analysis(episodes=50, max_steps=50):
    """Run training with price tracking and analysis."""
    print("\n=== Price Analysis Training Demo ===")
    
    # Create a simplified training environment
    config = {
        'market_volatility': 0.03,  # Reduce randomness for faster learning
        'max_steps': max_steps,     # Shorter episodes
        'variable_cost_per_unit': 50,  # Base cost is 50
        'competitor_price': 100     # Competitor price is 100
    }
    
    # Create environment
    env = BusinessEnvironment(
        config=config,
        max_steps=max_steps,
        log_path="../logs/price_analysis"
    )
    
    # Initialize custom tracking agent
    agent = PriceTrackingAgent(
        env=env,
        memory_size=10000,
        batch_size=64,
        gamma=0.95,
        epsilon=1.0,  # Start with full exploration
        epsilon_min=0.01,
        epsilon_decay=0.95,  # Faster decay for shorter training
        learning_rate=0.001,
        target_update_freq=10,
        checkpoint_dir="../models/price_analysis"
    )
    
    # Training loop
    print("Training agent and tracking price predictions...")
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Choose and take action
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Mark end of episode for analysis
        agent.mark_episode_end()
        
        # Print progress
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")
    
    # Analyze price predictions
    stats, price_counts = agent.analyze_price_distribution()
    
    print("\nPrice Prediction Analysis:")
    print(f"Mean Price: {stats['mean']:.2f}")
    print(f"Median Price: {stats['median']:.2f}")
    print(f"Min Price: {stats['min']:.2f}")
    print(f"Max Price: {stats['max']:.2f}")
    print(f"Standard Deviation: {stats['std']:.2f}")
    
    print("\nMost Common Prices:")
    for price, count in price_counts[:10]:  # Show top 10
        percentage = (count / len(agent.price_history)) * 100
        print(f"Price {price}: {count} occurrences ({percentage:.2f}%)")
    
    # Plot price evolution
    print("\nGenerating price evolution plots...")
    agent.plot_price_evolution(save_path="../logs/price_analysis/price_evolution.png")
    
    # Create a detailed price table by episode
    print("\nGenerating episode price table...")
    episode_data = []
    
    for i in range(len(agent.episode_boundaries) - 1):
        prices = agent.get_episode_prices(i)
        if prices:
            episode_data.append({
                'Episode': i + 1,
                'Min Price': np.min(prices),
                'Max Price': np.max(prices),
                'Mean Price': np.mean(prices),
                'Median Price': np.median(prices),
                'Price Variance': np.var(prices) if len(prices) > 1 else 0,
                'Unique Prices': len(set([round(p) for p in prices]))
            })
    
    # Convert to DataFrame and save
    episode_df = pd.DataFrame(episode_data)
    episode_df.to_csv("../logs/price_analysis/episode_price_analysis.csv", index=False)
    
    print(f"Analysis complete! Results saved to ../logs/price_analysis/")
    return agent


# Main execution
if __name__ == "__main__":
    print("BUSINESS SIMULATION PRICE ANALYSIS")
    print("==================================")
    
    # Run price analysis
    agent = run_price_analysis(episodes=50, max_steps=50)
    
    print("\nPrice analysis complete!")
    print("Check the logs directory for detailed results and visualizations.")