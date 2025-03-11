"""Pricing Strategy Module

This module contains different pricing strategies that can be used to determine
product pricing based on various business factors.
"""

from typing import Dict


class PricingStrategy:
    """Different pricing strategies to compare with the RL model recommendation."""
    
    @staticmethod
    def cost_plus_pricing(scenario_data: Dict[str, float], margin_pct: float = 50.0) -> Dict[str, float]:
        """Simple cost-plus pricing strategy."""
        unit_cost = scenario_data['unit_cost']
        price = unit_cost * (1 + margin_pct/100)
        return {'price': price}
    
    @staticmethod
    def competitor_match(scenario_data: Dict[str, float], discount_pct: float = 0.0) -> Dict[str, float]:
        """Match competitor price with optional discount."""
        competitor_price = scenario_data['competitor_price']
        price = competitor_price * (1 - discount_pct/100)
        return {'price': price}
    
    @staticmethod
    def premium_pricing(scenario_data: Dict[str, float], premium_pct: float = 15.0) -> Dict[str, float]:
        """Price higher than competitors to signal premium quality."""
        competitor_price = scenario_data['competitor_price']
        price = competitor_price * (1 + premium_pct/100)
        return {'price': price}
    
    @staticmethod
    def value_based_pricing(scenario_data: Dict[str, float]) -> Dict[str, float]:
        """Price based on perceived value, considering market conditions."""
        competitor_price = scenario_data['competitor_price']
        market_growth = scenario_data['market_growth']
        demand = scenario_data['demand']
        
        # Base value is competitor price
        base_price = competitor_price
        
        # Adjust based on market growth (growing markets can support higher prices)
        growth_factor = 1 + (market_growth * 2)  # e.g., 3% growth gives 6% price increase
        
        # Adjust based on demand (higher demand supports higher prices)
        demand_factor = 1 + (0.02 * (demand / 1000 - 4))  # Calibrated around 4000 demand
        
        price = base_price * growth_factor * demand_factor
        return {'price': price}
    
    @staticmethod
    def penetration_pricing(scenario_data: Dict[str, float], discount_pct: float = 20.0) -> Dict[str, float]:
        """Low initial price to gain market share."""
        competitor_price = scenario_data['competitor_price']
        price = competitor_price * (1 - discount_pct/100)
        return {'price': price}