#!/usr/bin/env python3
"""
Model Store Usage Example

This script demonstrates how to use the ModelStore class to properly save and load
models with architecture validation, preventing model architecture mismatches.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

# Import our ModelStore
from business_simulator.model_store import ModelStore

# Example model class
class SimpleNetwork(nn.Module):
    """A simple neural network for demonstration purposes."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Example model with different architecture
class DifferentNetwork(nn.Module):
    """A network with different architecture for testing mismatch detection."""
    
    def __init__(self, input_size=10, hidden_size=30, output_size=5):
        super(DifferentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def demonstrate_model_store():
    """Demonstrate the ModelStore functionality."""
    print("\n=== ModelStore Demonstration ===")
    
    # Create a model store
    store = ModelStore(base_dir="../models/demo_store")
    
    # Register model classes
    store.register_model_class("simple", SimpleNetwork)
    store.register_model_class("different", DifferentNetwork)
    
    # Create and save a model
    print("\nCreating and saving a model...")
    model = SimpleNetwork(input_size=10, hidden_size=20, output_size=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Save the model with metadata
    hyperparameters = {
        "input_size": 10,
        "hidden_size": 20,
        "output_size": 5,
        "learning_rate": 0.001
    }
    
    store.save(
        model=model,
        model_name="example_model",
        optimizer=optimizer,
        hyperparameters=hyperparameters,
        description="Example model for demonstration"
    )
    
    # List available models
    print("\nAvailable models:")
    models = store.list_models()
    for model_name, versions in models.items():
        print(f"  {model_name}: {', '.join(versions)}")
    
    # Load the model correctly
    print("\nLoading the model with correct architecture...")
    new_model = SimpleNetwork(input_size=10, hidden_size=20, output_size=5)
    new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    
    loaded_model, metadata = store.load(
        model_instance=new_model,
        model_name="example_model",
        optimizer=new_optimizer
    )
    
    print(f"Model loaded successfully!")
    print(f"Metadata: {metadata.get('description')}")
    
    # Demonstrate architecture mismatch detection
    print("\nDemonstrating architecture mismatch detection...")
    try:
        # Try to load with a different architecture
        wrong_model = SimpleNetwork(input_size=10, hidden_size=30, output_size=5)  # Different hidden size
        store.load(wrong_model, "example_model", strict=True)
    except ValueError as e:
        print(f"Correctly detected architecture mismatch: {str(e)}")
    
    # Load with strict=False
    print("\nLoading with strict=False (allowing mismatch)...")
    try:
        wrong_model = SimpleNetwork(input_size=10, hidden_size=30, output_size=5)
        loaded_model, _ = store.load(wrong_model, "example_model", strict=False)
        print("Loaded despite architecture mismatch (with warning)")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Load by type
    print("\nLoading model by registered type...")
    try:
        model_by_type, _ = store.load_by_type(
            model_type="simple",
            model_name="example_model",
            input_size=10,
            hidden_size=20,
            output_size=5
        )
        print("Successfully loaded model by type")
    except Exception as e:
        print(f"Error loading by type: {str(e)}")
    
    # Get metadata without loading
    print("\nGetting metadata without loading the model...")
    metadata = store.get_metadata("example_model")
    print(f"Model description: {metadata.get('description')}")
    print(f"Model type: {metadata.get('model_type')}")
    
    return store

def integrate_with_dqn_agent():
    """Example of how to integrate ModelStore with DQNAgent."""
    print("\n=== Integration with DQNAgent Example ===")
    
    # Import DQNAgent and related modules
    from business_simulator.business_environment import BusinessEnvironment
    from business_simulator.reinforcement_learning_agent import DQNAgent, DQNetwork
    
    # Create a model store
    store = ModelStore(base_dir="../models/dqn_store")
    
    # Register DQNetwork class
    store.register_model_class("dqn_network", DQNetwork)
    
    # Create environment and agent
    env = BusinessEnvironment()
    agent = DQNAgent(env=env)
    
    # Save the agent's model
    print("\nSaving DQN model...")
    hyperparameters = {
        "memory_size": agent.memory_size,
        "batch_size": agent.batch_size,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "learning_rate": agent.learning_rate
    }
    
    store.save(
        model=agent.model,
        model_name="dqn_example",
        optimizer=agent.optimizer,
        hyperparameters=hyperparameters,
        training_steps=agent.train_step
    )
    
    # Create a new agent
    new_agent = DQNAgent(env=env)
    
    # Load the model into the new agent
    print("\nLoading DQN model into new agent...")
    loaded_model, metadata = store.load(
        model_instance=new_agent.model,
        model_name="dqn_example",
        optimizer=new_agent.optimizer
    )
    
    # Update the agent's model
    new_agent.model = loaded_model
    new_agent.update_target_network()
    
    print(f"Successfully loaded DQN model")
    if 'hyperparameters' in metadata:
        print(f"Loaded hyperparameters: {metadata['hyperparameters']}")
    
    return new_agent

def modify_business_advisor_to_use_model_store():
    """Example of how to modify BusinessAdvisor to use ModelStore."""
    print("\n=== Modified BusinessAdvisor Example ===")
    
    # This is an example of how the BusinessAdvisor.__init__ method could be modified
    # to use ModelStore for safer model loading
    
    code_example = """
    # In business_advisor.py:
    
    from business_simulator.model_store import ModelStore
    
    class BusinessAdvisor:
        def __init__(self, model_path, config=None):
            self.model_path = model_path
            self.config = config if config else {}
            
            # Create a reference environment
            self.env = BusinessEnvironment(config=config)
            
            # Use ModelStore for safe model loading
            store = ModelStore()
            
            # Create agent with proper architecture
            self.agent = DQNAgent(env=self.env)
            
            # Load the model with architecture validation
            try:
                self.agent.model, metadata = store.load(
                    model_instance=self.agent.model,
                    model_name=os.path.basename(model_path),
                    optimizer=self.agent.optimizer,
                    strict=True  # Enforce architecture matching
                )
                
                # Update target network
                self.agent.update_target_network()
                
                # Set to evaluation mode
                self.agent.epsilon = 0.01
                
                print(f"Successfully loaded model with architecture validation")
                
            except ValueError as e:
                raise ValueError(f"Model architecture mismatch: {str(e)}. The model structure may have changed since training.")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model file not found: {str(e)}")
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
    """
    
    print("Example code for modifying BusinessAdvisor:")
    print(code_example)

# Main execution
if __name__ == "__main__":
    print("MODEL STORE USAGE EXAMPLES")
    print("==========================")
    
    # Run the demonstrations
    store = demonstrate_model_store()
    
    # Show integration with DQNAgent
    agent = integrate_with_dqn_agent()
    
    # Show how to modify BusinessAdvisor
    modify_business_advisor_to_use_model_store()
    
    print("\nDemonstration complete!")
    print("The ModelStore class provides a robust solution for managing models")
    print("and preventing architecture mismatches during loading.")