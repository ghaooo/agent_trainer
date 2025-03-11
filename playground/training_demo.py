#!/usr/bin/env python3
"""Training Demo for Business Simulator

This script demonstrates how to train a reinforcement learning agent using the business_simulator package.
It shows how to create a training environment, initialize a DQN agent, and train it to make optimal business decisions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the business_simulator package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from business_simulator.business_environment import BusinessEnvironment
from business_simulator.reinforcement_learning_agent import DQNAgent
from business_simulator.train_and_evaluate import train_dqn_agent, evaluate_agent

# Set matplotlib to non-interactive mode to prevent blocking
plt.ioff()

# Create output directories


def run_quick_training(episodes=100):
    """Run a quick training demonstration."""
    print("\n=== Quick RL Agent Training Demo ===")
    
    # Create a simplified training environment with more stable parameters
    config = {
        'market_volatility': 0.02,  # Further reduce randomness for more stable learning
        'max_steps': 50,           # Shorter episodes
        'market_size': 10000,      # Set reasonable market size
        'initial_cash': 100000,    # Set reasonable initial cash
        'variable_cost_per_unit': 50,  # Set base cost
        'competitor_price': 100,    # Set competitor price benchmark
        'market_growth_rate': 0.01  # Slow, stable growth
    }
    
    try:
        # Create environment
        env = BusinessEnvironment(
            config=config,
            max_steps=50,
            log_path="../logs/training_demo"
        )
        
        # Create a new agent for training with adjusted parameters
        agent = DQNAgent(
            env=env,
            memory_size=10000,
            batch_size=64,
            gamma=0.99,        # Higher gamma for better long-term planning
            epsilon=1.0,
            epsilon_min=0.1,   # Higher minimum exploration
            epsilon_decay=0.995,  # Slower decay for better exploration
            learning_rate=0.001
        )
        
        # Train the agent
        agent, stats = train_dqn_agent(
            env_config=config,
            episodes=episodes,
            max_steps=50,
            save_freq=50,
            eval_freq=25,
            log_dir="../logs/training_demo",
            model_dir="./",
            project_name='business-simulator',
            experiment_name=f'training_demo_{episodes}_episodes'
        )
        
        print("Training complete!")
        results = None
        
        return agent, stats, results
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        if 'agent' in locals():
            agent.save(os.path.join("../models/demo", "dqn_model_interrupted"))
            print("Model saved to: ../models/demo/dqn_model_interrupted")
        raise
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

# Main execution
if __name__ == "__main__":
    print("BUSINESS SIMULATION TRAINING DEMO")
    print("================================")
    
    # Set number of training episodes
    training_episodes = 200  # Increase from 20 to 200
    
    # Run training demo
    agent, stats, results = run_quick_training(episodes=training_episodes)
    
    print("\nTraining demonstration complete!")
    print(f"Model saved to: ../models/demo/dqn_model_final")
    print("\nTry running the business_advisor_demo.py script to see how to use the trained model!")