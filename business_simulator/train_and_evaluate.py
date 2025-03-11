import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import time
import wandb

# Set matplotlib to non-interactive mode
plt.ioff()

# Import our business environment and agents
from business_simulator.business_environment import BusinessEnvironment
from business_simulator.agents import RandomAgent, SimplePolicyAgent
from business_simulator.simulation import run_simulation
from business_simulator.reinforcement_learning_agent import DQNAgent

def train_dqn_agent(
    env_config=None,
    episodes=1000,
    max_steps=100,
    save_freq=100,
    log_dir='logs/training',
    model_dir='models',
    render_freq=None,
    eval_freq=50,
    agent=None,  # Added parameter to accept an existing agent
    project_name='business-simulator',  # wandb project name
    experiment_name=None  # wandb run name
):
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            'env_config': env_config,
            'episodes': episodes,
            'max_steps': max_steps,
            'save_freq': save_freq,
            'eval_freq': eval_freq,
            'agent_params': {
                'memory_size': 10000,
                'batch_size': 64,
                'gamma': 0.95,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'learning_rate': 0.001,
                'target_update_freq': 10
            }
        }
    )

    """
    Train a DQN agent in the business environment.
    
    Args:
        env_config: Configuration for the business environment
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        save_freq: Frequency to save model checkpoints
        log_dir: Directory to save training logs
        model_dir: Directory to save model checkpoints
        render_freq: Frequency to render the environment (None = no rendering)
        eval_freq: Frequency to evaluate the agent's performance
        agent: An existing agent to continue training (None = create new agent)
        
    Returns:
        agent: The trained DQN agent
        training_stats: Dictionary with training statistics
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    env = BusinessEnvironment(
        config=env_config,
        max_steps=max_steps,
        log_path=log_dir
    )
    
    # Create evaluation environment (separate from training)
    eval_env = BusinessEnvironment(
        config=env_config,
        max_steps=max_steps,
        log_path=log_dir
    )
    
    # Create agent if not provided
    if agent is None:
        agent = DQNAgent(
            env=env,
            memory_size=10000,
            batch_size=64,
            gamma=0.99,  # Increase gamma to value future rewards more
            epsilon=1.0,
            epsilon_min=0.1,  # Increase minimum exploration rate
            epsilon_decay=0.998,  # Slower decay for more exploration
            learning_rate=0.001,
            target_update_freq=10,
            checkpoint_dir=model_dir
        )
    else:
        # Update the agent's environment to the current one
        agent.env = env
    
    # Create baseline agents for comparison
    random_agent = RandomAgent(eval_env)
    heuristic_agent = SimplePolicyAgent(eval_env)
    
    # Training statistics
    stats = {
        'episode_rewards': [],
        'episode_profits': [],
        'episode_cash': [],
        'episode_steps': [],
        'eval_rewards': [],
        'eval_profits': [],
        'eval_cash': [],
        'baseline_random_rewards': [],
        'baseline_heuristic_rewards': []
    }
    
    # Training loop
    for episode in tqdm(range(episodes), desc="Training DQN Agent"):
        # Reset environment
        state, info = env.reset()
        
        total_reward = 0
        episode_profits = []
        
        # Run one episode
        for step in range(max_steps):
            # Choose and take action
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience and learn
            agent.learn(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            episode_profits.append(info.get('profit', 0))
            
            # Render if requested
            if render_freq is not None and episode % render_freq == 0:
                env.render()
            
            if done:
                break
        
        # Log episode metrics to wandb
        wandb.log({
            'episode': episode,
            'reward': total_reward,
            'profit': sum(episode_profits),
            'cash': env.cash,
            'steps': env.step_count,
            'epsilon': agent.epsilon
        })
        
        # Evaluate periodically
        if episode % eval_freq == 0 or episode == episodes - 1:
            # Set to evaluation mode (low epsilon)
            original_epsilon = agent.epsilon
            agent.epsilon = 0.05
            
            # Run evaluation episode
            eval_results = run_simulation(eval_env, agent, episodes=5, render=False)
            random_results = run_simulation(eval_env, random_agent, episodes=5, render=False)
            heuristic_results = run_simulation(eval_env, heuristic_agent, episodes=5, render=False)
            
            # Log evaluation metrics to wandb
            wandb.log({
                'eval/reward': np.mean(eval_results['episode_rewards']),
                'eval/profit': np.mean(eval_results['episode_profits']),
                'eval/cash': np.mean(eval_results['episode_cash']),
                'baseline/random_reward': np.mean(random_results['episode_rewards']),
                'baseline/heuristic_reward': np.mean(heuristic_results['episode_rewards'])
            })
            
            # Print evaluation summary
            print(f"\nEvaluation at episode {episode}:")
            print(f"  DQN Agent: Reward={np.mean(eval_results['episode_rewards']):.2f}, Cash={np.mean(eval_results['episode_cash']):.2f}")
            print(f"  Random Agent: Reward={np.mean(random_results['episode_rewards']):.2f}")
            print(f"  Heuristic Agent: Reward={np.mean(heuristic_results['episode_rewards']):.2f}")
            
            # Restore training epsilon
            agent.epsilon = original_epsilon
        
        # Save model periodically
        if episode % save_freq == 0 or episode == episodes - 1:
            model_path = os.path.join(model_dir, f"dqn_model_episode_{episode}")
            agent.save(model_path)
            wandb.save(os.path.basename(f"{model_path}.pt"))
    
    # Save final model and finish wandb run
    final_model_path = agent.save(os.path.join(model_dir, "dqn_model_final"))
    wandb.save(os.path.basename(f"{final_model_path}.pt"))
    wandb.finish()
    
    return agent, None  # Return None instead of stats since we're using wandb

def plot_training_progress(stats, log_dir):
    """Plot training progress metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(stats['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot episode profits
    plt.subplot(2, 2, 2)
    plt.plot(stats['episode_profits'])
    plt.title('Episode Profits')
    plt.xlabel('Episode')
    plt.ylabel('Total Profit')
    
    # Plot final cash
    plt.subplot(2, 2, 3)
    plt.plot(stats['episode_cash'])
    plt.title('Final Cash')
    plt.xlabel('Episode')
    plt.ylabel('Cash')
    
    # Plot evaluation comparison
    if len(stats['eval_rewards']) > 0:
        plt.subplot(2, 2, 4)
        
        eval_episodes = list(range(0, len(stats['episode_rewards']), 50))
        if len(eval_episodes) != len(stats['eval_rewards']):
            eval_episodes = list(range(0, len(stats['episode_rewards']), len(stats['episode_rewards']) // len(stats['eval_rewards'])))
        
        plt.plot(eval_episodes, stats['eval_rewards'], 'b-', label='DQN Agent')
        plt.plot(eval_episodes, stats['baseline_random_rewards'], 'r-', label='Random Agent')
        plt.plot(eval_episodes, stats['baseline_heuristic_rewards'], 'g-', label='Heuristic Agent')
        plt.title('Agent Comparison (Evaluation)')
        plt.xlabel('Training Episode')
        plt.ylabel('Average Reward')
        plt.legend()
    
    plt.tight_layout()
    # Ensure logs directory exists and save the figure
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, 'training_progress.png'))
    plt.close()

def evaluate_agent(
    agent,
    env_config=None,
    episodes=100,
    max_steps=100,
    render=False,
    log_dir='logs/evaluation'
):
    """
    Evaluate a trained agent in various business scenarios.
    
    Args:
        agent: The trained agent
        env_config: Configuration for the business environment
        episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        log_dir: Directory to save evaluation logs
        
    Returns:
        results: Dictionary with evaluation results
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = BusinessEnvironment(
        config=env_config,
        max_steps=max_steps,
        log_path=log_dir,
        render_mode="human" if render else None
    )
    
    # Create baseline agents for comparison
    random_agent = RandomAgent(env)
    heuristic_agent = SimplePolicyAgent(env)
    
    # Evaluate trained agent
    print("Evaluating DQN Agent...")
    dqn_results = run_simulation(env, agent, episodes=episodes, render=render)
    
    # Evaluate random agent
    print("Evaluating Random Agent...")
    random_results = run_simulation(env, random_agent, episodes=episodes, render=render)
    
    # Evaluate heuristic agent
    print("Evaluating Heuristic Agent...")
    heuristic_results = run_simulation(env, heuristic_agent, episodes=episodes, render=render)
    
    # Combine results
    results = {
        'dqn': dqn_results,
        'random': random_results,
        'heuristic': heuristic_results
    }
    
    # Plot comparison
    plot_agent_comparison(results, log_dir)
    
    # Save results
    results_path = os.path.join(log_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        # Ensure all values are Python natives (not numpy)
        results_json = {
            agent_type: {
                k: [float(val) for val in v] if isinstance(v, list) else v
                for k, v in agent_results.items()
            }
            for agent_type, agent_results in results.items()
        }
        json.dump(results_json, f)
    
    return results

def plot_agent_comparison(results, log_dir):
    """Plot comparison of different agents."""
    plt.figure(figsize=(15, 5))
    
    # Calculate statistics
    avg_rewards = {
        'DQN Agent': np.mean(results['dqn']['episode_rewards']),
        'Random Agent': np.mean(results['random']['episode_rewards']),
        'Heuristic Agent': np.mean(results['heuristic']['episode_rewards'])
    }
    
    avg_profits = {
        'DQN Agent': np.mean(results['dqn']['episode_profits']),
        'Random Agent': np.mean(results['random']['episode_profits']),
        'Heuristic Agent': np.mean(results['heuristic']['episode_profits'])
    }
    
    avg_cash = {
        'DQN Agent': np.mean(results['dqn']['episode_cash']),
        'Random Agent': np.mean(results['random']['episode_cash']),
        'Heuristic Agent': np.mean(results['heuristic']['episode_cash'])
    }
    
    # Plot average rewards
    plt.subplot(1, 3, 1)
    agents = list(avg_rewards.keys())
    values = list(avg_rewards.values())
    plt.bar(agents, values)
    plt.title('Average Reward')
    plt.ylabel('Reward')
    plt.xticks(rotation=45)
    
    # Plot average profits
    plt.subplot(1, 3, 2)
    values = list(avg_profits.values())
    plt.bar(agents, values)
    plt.title('Average Profit')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)
    
    # Plot average final cash
    plt.subplot(1, 3, 3)
    values = list(avg_cash.values())
    plt.bar(agents, values)
    plt.title('Average Final Cash')
    plt.ylabel('Cash')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    # Ensure logs directory exists and save the figure
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, 'agent_comparison.png'))
    plt.close()

def create_business_decision_agent(model_path, config=None):
    """
    Create a business decision agent using a trained model.
    
    Args:
        model_path: Path to the trained model
        config: Configuration for the business environment
        
    Returns:
        agent: The business decision agent
    """
    # Create a temporary environment to initialize the agent
    env = BusinessEnvironment(config=config)
    
    # Create agent
    agent = DQNAgent(env=env)
    
    # Load model
    agent.load(model_path)
    
    # Set to evaluation mode (low epsilon)
    agent.epsilon = 0.01
    
    return agent