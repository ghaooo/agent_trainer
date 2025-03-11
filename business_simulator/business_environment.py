import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import random
import json
import os

class BusinessEnvironment(gym.Env):
    """
    A simple business environment for reinforcement learning that simulates
    basic business operations with customizable parameters.
    
    This environment simulates a business making pricing, marketing, and inventory
    decisions to maximize profit over time.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 100,
        log_path: Optional[str] = None
    ):
        """
        Initialize the business environment with configurable parameters.
        
        Args:
            config: Dictionary of configuration parameters
            render_mode: How to render the environment
            max_steps: Maximum number of steps per episode
            log_path: Path to save logs and simulation data
        """
        # Default configuration if none provided
        default_config = {
            # Market parameters
            'market_size': 10000,           # Total potential customers
            'price_elasticity': -1.5,       # Price elasticity of demand
            'marketing_effectiveness': 0.2,  # Impact of marketing on demand
            'competitor_price': 100,        # Average competitor price
            'market_growth_rate': 0.01,     # Base market growth per step
            'market_volatility': 0.05,      # Random fluctuations in market
            
            # Business parameters
            'initial_cash': 100000,         # Starting cash
            'fixed_costs': 5000,            # Costs per step regardless of sales
            'variable_cost_per_unit': 50,   # Production cost per unit
            'inventory_holding_cost': 2,    # Cost per unit in inventory
            'stockout_penalty': 10,         # Penalty per unsatisfied demand
            'initial_inventory': 200,       # Starting inventory
            'max_production': 500,          # Maximum production per step
            'max_marketing_budget': 10000,  # Maximum marketing spend
            
            # Environment parameters
            'random_events_enabled': False, # Whether random events occur
            'seasonal_effects_enabled': False, # Whether seasonal patterns exist
        }
        
        # Update config with user-provided values
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        self.max_steps = max_steps
        self.step_count = 0
        self.render_mode = render_mode
        self.log_path = log_path
        
        # Create logging directory if specified
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)
            
        # Initialize history storage
        self.history = {
            'state': [],
            'action': [],
            'reward': [],
            'metrics': []
        }
        
        # Define action space: [price, production_qty, marketing_budget]
        self.action_space = spaces.Box(
            low=np.array([
                self.config['variable_cost_per_unit'] * 1.1,  # Minimum price
                0,                                            # Minimum production
                0                                             # Minimum marketing
            ], dtype=np.float32),
            high=np.array([
                self.config['competitor_price'] * 1.5,        # Maximum price
                self.config['max_production'],                # Maximum production
                self.config['max_marketing_budget']           # Maximum marketing
            ], dtype=np.float32)
        )
        
        # Define observation space (state space)
        self.observation_space = spaces.Box(
            low=np.array([
                0,                               # Cash (can't go below 0)
                0,                               # Inventory
                0,                               # Market demand
                self.config['variable_cost_per_unit'],  # Min cost
                self.config['competitor_price'] * 0.5,  # Min competitor price
                0                                # Market growth (can be negative)
            ], dtype=np.float32),
            high=np.array([
                float('inf'),                    # Cash (no upper limit)
                float('inf'),                    # Inventory (no practical upper limit)
                self.config['market_size'] * 2,  # Maximum possible demand
                self.config['variable_cost_per_unit'] * 2,  # Max cost
                self.config['competitor_price'] * 1.5,       # Max competitor price
                0.2                              # Max market growth
            ], dtype=np.float32)
        )
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Reset business state
        self.cash = self.config['initial_cash']
        self.inventory = self.config['initial_inventory']
        self.current_demand = self.config['market_size'] * 0.1  # Start with 10% market
        self.current_cost = self.config['variable_cost_per_unit']
        self.competitor_price = self.config['competitor_price']
        self.market_growth = self.config['market_growth_rate']
        
        # Reset history
        self.history = {
            'state': [],
            'action': [],
            'reward': [],
            'metrics': []
        }
        
        # Get observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a business decision step in the environment.
        
        Args:
            action: [price, production_qty, marketing_budget]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Increment step counter
        self.step_count += 1
        
        # Extract actions
        price, production_qty, marketing_budget = action
        
        # Ensure actions are within bounds
        price = max(self.current_cost * 1.1, min(price, self.competitor_price * 1.5))
        production_qty = max(0, min(production_qty, self.config['max_production']))
        marketing_budget = max(0, min(marketing_budget, self.config['max_marketing_budget']))
        
        # Update inventory with new production
        self.inventory += production_qty
        
        # Calculate production costs
        production_cost = production_qty * self.current_cost
        
        # Update cash with production costs and fixed costs
        self.cash -= production_cost + self.config['fixed_costs']
        
        # Calculate expected demand based on price and marketing
        price_effect = (self.competitor_price / price) ** self.config['price_elasticity']
        marketing_effect = 1 + self.config['marketing_effectiveness'] * (marketing_budget / self.config['max_marketing_budget'])
        
        # Apply market growth and random fluctuations
        market_factor = 1 + self.market_growth + np.random.normal(0, self.config['market_volatility'])
        market_factor = max(0.8, min(market_factor, 1.2))  # Limit volatility impact
        
        # Calculate actual demand
        actual_demand = int(self.current_demand * price_effect * marketing_effect * market_factor)
        
        # Calculate sales (limited by inventory)
        sales = min(self.inventory, actual_demand)
        unfulfilled_demand = actual_demand - sales
        
        # Update inventory and cash based on sales
        self.inventory -= sales
        revenue = sales * price
        self.cash += revenue
        
        # Apply inventory holding cost
        holding_cost = self.inventory * self.config['inventory_holding_cost']
        self.cash -= holding_cost
        
        # Apply stockout penalty
        stockout_penalty = unfulfilled_demand * self.config['stockout_penalty']
        self.cash -= stockout_penalty
        
        # Update market variables for next step
        self.current_demand = actual_demand
        self.market_growth = self.config['market_growth_rate'] + np.random.normal(0, 0.005)
        self.competitor_price = self.config['competitor_price'] * (1 + np.random.normal(0, 0.02))
        self.current_cost = self.config['variable_cost_per_unit'] * (1 + np.random.normal(0, 0.01))
        
        # Calculate metrics for this step
        profit = revenue - production_cost - self.config['fixed_costs'] - holding_cost - stockout_penalty - marketing_budget
        
        # Calculate reward (profit)
        reward = profit
        
        # Check if business is bankrupt
        bankrupt = self.cash <= 0
        
        # Check if episode is done
        terminated = bankrupt or self.step_count >= self.max_steps
        truncated = False
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add additional metrics to info
        info.update({
            'profit': profit,
            'revenue': revenue,
            'production_cost': production_cost,
            'fixed_costs': self.config['fixed_costs'],
            'holding_cost': holding_cost,
            'stockout_penalty': stockout_penalty,
            'marketing_budget': marketing_budget,
            'actual_demand': actual_demand,
            'sales': sales,
            'unfulfilled_demand': unfulfilled_demand,
            'bankrupt': int(bankrupt)  # Convert boolean to int for JSON serialization
        })
        
        # Store history
        self.history['state'].append(observation.tolist())
        self.history['action'].append([price, production_qty, marketing_budget])
        self.history['reward'].append(reward)
        self.history['metrics'].append(info)
        
        # Return step results
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Return the current state observation."""
        return np.array([
            self.cash,
            self.inventory,
            self.current_demand,
            self.current_cost,
            self.competitor_price,
            self.market_growth
        ], dtype=np.float32)
    
    def _get_info(self):
        """Return additional information for monitoring."""
        return {
            'step': self.step_count,
            'cash': self.cash,
            'inventory': self.inventory,
            'demand': self.current_demand,
            'cost': self.current_cost,
            'competitor_price': self.competitor_price,
            'market_growth': self.market_growth
        }
    
    def render(self):
        """Render the environment visualization."""
        if self.render_mode == "human" and len(self.history['metrics']) > 0:
            # Simple visualization of key metrics
            plt.figure(figsize=(15, 10))
            
            # Extract metrics for plotting
            steps = list(range(1, len(self.history['metrics']) + 1))
            cash = [m['cash'] for m in self.history['metrics']]
            inventory = [m['inventory'] for m in self.history['metrics']]
            demand = [m['demand'] for m in self.history['metrics']]
            profits = [m.get('profit', 0) for m in self.history['metrics']]
            
            # Plot cash and profits
            plt.subplot(2, 2, 1)
            plt.plot(steps, cash, 'b-', label='Cash')
            plt.plot(steps, profits, 'g-', label='Profit')
            plt.title('Cash and Profit')
            plt.xlabel('Steps')
            plt.legend()
            
            # Plot inventory and demand
            plt.subplot(2, 2, 2)
            plt.plot(steps, inventory, 'r-', label='Inventory')
            plt.plot(steps, demand, 'k-', label='Demand')
            plt.title('Inventory and Demand')
            plt.xlabel('Steps')
            plt.legend()
            
            # Plot actions if available
            if len(self.history['action']) > 0:
                prices = [a[0] for a in self.history['action']]
                production = [a[1] for a in self.history['action']]
                marketing = [a[2] for a in self.history['action']]
                
                plt.subplot(2, 2, 3)
                plt.plot(steps, prices, 'm-', label='Price')
                plt.title('Pricing Decisions')
                plt.xlabel('Steps')
                
                plt.subplot(2, 2, 4)
                plt.plot(steps, production, 'c-', label='Production')
                plt.plot(steps, marketing, 'y-', label='Marketing (x10)')
                plt.title('Production and Marketing Decisions')
                plt.xlabel('Steps')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        # Return RGB array for other rendering modes
        return None
    
    def save_history(self, filename=None):
        """Save simulation history to a file."""
        if self.log_path:
            if filename is None:
                filename = f"simulation_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert numpy types to native Python types for JSON serialization
            serializable_history = {
                'state': [[float(val) for val in state] for state in self.history['state']],
                'action': [[float(val) for val in action] for action in self.history['action']],
                'reward': [float(reward) for reward in self.history['reward']],
                'metrics': [{
                    k: float(v) if isinstance(v, (np.number, np.float32, np.float64, np.int32, np.int64)) else v
                    for k, v in metric.items()
                } for metric in self.history['metrics']]
            }
            
            filepath = os.path.join(self.log_path, filename)
            with open(filepath, 'w') as f:
                json.dump(serializable_history, f)
            
            return filepath
        return None

    def close(self):
        """Clean up resources."""
        if self.render_mode == "human":
            plt.close('all')