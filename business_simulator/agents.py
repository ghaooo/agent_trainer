import numpy as np
from typing import Dict, List, Tuple, Any

class RandomAgent:
    """
    A simple random agent that takes random actions within the environment's action space.
    Used for testing and as a baseline.
    """
    def __init__(self, env):
        self.env = env
    
    def act(self, observation):
        """Take a random action."""
        return self.env.action_space.sample()
    
    def learn(self, *args, **kwargs):
        """Placeholder for learning method."""
        pass


class SimplePolicyAgent:
    """
    A simple rule-based agent that follows basic business heuristics.
    Used as a baseline comparison for learning agents.
    """
    def __init__(self, env):
        self.env = env
        self.price_margin = 0.3  # Target 30% profit margin
    
    def act(self, observation):
        """Take a heuristic-based action."""
        cash, inventory, demand, cost, competitor_price, market_growth = observation
        
        # Set price based on cost plus margin, and competitor price
        target_price = cost * (1 + self.price_margin)
        # Adjust price based on competitor and inventory levels
        if inventory < demand * 0.5:  # Low inventory
            price = max(target_price, competitor_price * 0.95)
        elif inventory > demand * 2:  # High inventory
            price = min(target_price * 0.9, competitor_price * 0.9)
        else:  # Normal inventory
            price = min(target_price, competitor_price * 0.97)
        
        # Set production based on demand and inventory
        ideal_inventory = demand * 1.5
        if inventory < ideal_inventory:
            production_qty = min(ideal_inventory - inventory, self.env.config['max_production'])
        else:
            production_qty = 0
        
        # Set marketing budget based on cash and market conditions
        if market_growth < 0:  # Declining market
            marketing_pct = 0.1  # Increase marketing to 10% of cash
        else:  # Growing market
            marketing_pct = 0.05  # Normal marketing at 5% of cash
        
        marketing_budget = min(cash * marketing_pct, self.env.config['max_marketing_budget'])
        
        return np.array([price, production_qty, marketing_budget])
    
    def learn(self, *args, **kwargs):
        """Placeholder for learning method."""
        pass