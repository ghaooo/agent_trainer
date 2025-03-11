#!/usr/bin/env python3
"""
Basic Business Simulation Demo

This script demonstrates how to run a basic business simulation using the business_simulator package.
It shows how to create a business environment, initialize agents, and run simulations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the business_simulator package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from business_simulator.business_environment import BusinessEnvironment
from business_simulator.agents import RandomAgent, SimplePolicyAgent
from business_simulator.simulation import run_simulation

# Create output directories
os.makedirs("../logs", exist_ok=True)

def run_basic_simulation():
    """Run a basic simulation with the heuristic agent."""
    print("\n=== Basic Simulation Demo ===")
    
    # Create environment
    env = BusinessEnvironment(
        render_mode="human",
        max_steps=50,
        log_path="../logs/basic_demo"
    )
    
    # Create simple rule-based agent
    agent = SimplePolicyAgent(env)
    
    # Run simulation
    results = run_simulation(env, agent, episodes=1, render=True)
    
    # Print results
    print(f"Episode reward: {results['episode_rewards'][0]:.2f}")
    print(f"Final cash: ${results['episode_cash'][0]:.2f}")
    print(f"Simulation completed in {results['episode_steps'][0]} steps")
    
    return env, agent, results

def run_custom_business_scenario():
    """Run a simulation with a custom business scenario."""
    print("\n=== Custom Business Scenario Demo ===")
    
    # Define custom business scenario
    custom_config = {
        # Market parameters
        'market_size': 5000,            # Smaller market
        'price_elasticity': -2.0,       # More price sensitive
        'marketing_effectiveness': 0.3,  # Better marketing response
        'competitor_price': 120,         # Higher competitor price
        'market_growth_rate': 0.02,      # Faster growth
        'market_volatility': 0.08,       # More volatile
        
        # Business parameters
        'initial_cash': 80000,           # Less starting cash
        'fixed_costs': 8000,             # Higher fixed costs
        'variable_cost_per_unit': 70,    # Higher production cost
        'inventory_holding_cost': 3,     # Higher inventory cost
        'stockout_penalty': 15,          # Higher stockout penalty
        'initial_inventory': 100,        # Less starting inventory
        'max_production': 300,           # Lower max production
        'max_marketing_budget': 15000,   # Higher max marketing
    }
    
    # Create environment with custom config
    env = BusinessEnvironment(
        config=custom_config,
        render_mode="human",
        max_steps=50,
        log_path="../logs/custom_demo"
    )
    
    # Compare random vs heuristic agent
    random_agent = RandomAgent(env)
    heuristic_agent = SimplePolicyAgent(env)
    
    # Run simulations
    print("Running random agent simulation...")
    random_results = run_simulation(env, random_agent, episodes=1, render=True)
    
    print("Running heuristic agent simulation...")
    heuristic_results = run_simulation(env, heuristic_agent, episodes=1, render=True)
    
    # Print comparison
    print("\nRandom Agent:")
    print(f"  Reward: {random_results['episode_rewards'][0]:.2f}")
    print(f"  Final Cash: ${random_results['episode_cash'][0]:.2f}")
    
    print("\nHeuristic Agent:")
    print(f"  Reward: {heuristic_results['episode_rewards'][0]:.2f}")
    print(f"  Final Cash: ${heuristic_results['episode_cash'][0]:.2f}")
    
    return env, custom_config

# Main execution
if __name__ == "__main__":
    print("BUSINESS SIMULATION SANDBOX DEMO")
    print("================================")
    
    # Run basic simulation
    env, agent, results = run_basic_simulation()
    
    # Run custom scenario
    env, custom_config = run_custom_business_scenario()
    
    print("\nDemonstration complete!")