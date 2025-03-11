import numpy as np
import os
import json

def run_simulation(env, agent, episodes=1, render=False):
    """
    Run a complete simulation with the given agent and environment.
    
    Args:
        env: The business environment
        agent: The decision-making agent
        episodes: Number of episodes to run
        render: Whether to render the environment
    
    Returns:
        results: Dictionary of simulation results
    """
    results = {
        'episode_rewards': [],
        'episode_profits': [],
        'episode_steps': [],
        'episode_cash': []
    }
    
    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_profits = []
        done = False
        
        while not done:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Learning step for agent (if implemented)
            agent.learn(observation, action, reward, next_observation, done)
            
            observation = next_observation
            episode_reward += reward
            episode_profits.append(info.get('profit', 0))
            
            if render:
                env.render()
        
        # Save episode results
        results['episode_rewards'].append(episode_reward)
        results['episode_profits'].append(sum(episode_profits))
        results['episode_steps'].append(env.step_count)
        results['episode_cash'].append(env.cash)
        
        # Save history to file if log_path is specified
        if env.log_path:
            env.save_history(f"simulation_episode_{episode}.json")
    
    return results