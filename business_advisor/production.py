"""Production Strategy Module

This module contains different production strategies that can be used to determine
production quantities based on various business factors.
"""

from typing import Dict


class ProductionStrategy:
    """Different production strategies to compare with the RL model recommendation."""
    
    @staticmethod
    def demand_matching(scenario_data: Dict[str, float], buffer_pct: float = 10.0) -> Dict[str, float]:
        """Match production to current demand with buffer."""
        demand = scenario_data['demand']
        production = demand * (1 + buffer_pct/100)
        return {'production_quantity': production}
    
    @staticmethod
    def inventory_balancing(scenario_data: Dict[str, float], target_months: float = 3.0) -> Dict[str, float]:
        """Produce to maintain target months of inventory."""
        demand = scenario_data['demand']
        current_inventory = scenario_data['inventory']
        monthly_demand = demand / 12  # Assuming demand is annual
        
        target_inventory = monthly_demand * target_months
        production_needed = target_inventory - current_inventory
        
        # Ensure production is not negative
        production = max(0, production_needed)
        return {'production_quantity': production}
    
    @staticmethod
    def growth_anticipation(scenario_data: Dict[str, float]) -> Dict[str, float]:
        """Adjust production based on anticipated market growth."""
        demand = scenario_data['demand']
        market_growth = scenario_data['market_growth']
        current_inventory = scenario_data['inventory']
        
        # Project future demand
        projected_demand = demand * (1 + market_growth)
        
        # Account for existing inventory
        adjusted_production = projected_demand - current_inventory
        production = max(0, adjusted_production)
        return {'production_quantity': production}