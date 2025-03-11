"""Marketing Strategy Module

This module contains different marketing strategies that can be used to determine
marketing budget allocations based on various business factors.
"""

from typing import Dict


class MarketingStrategy:
    """Different marketing strategies to compare with the RL model recommendation."""
    
    @staticmethod
    def percentage_of_revenue(scenario_data: Dict[str, float], price: float, production: float, 
                             percent: float = 15.0) -> Dict[str, float]:
        """Set marketing budget as percentage of expected revenue."""
        expected_revenue = min(production, scenario_data['demand']) * price
        marketing_budget = expected_revenue * (percent / 100)
        return {'marketing_budget': marketing_budget}
    
    @staticmethod
    def competitive_parity(scenario_data: Dict[str, float]) -> Dict[str, float]:
        """Estimate and match competitor's marketing spending."""
        # Assume competitor spending is related to their price and market demand
        competitor_price = scenario_data['competitor_price']
        demand = scenario_data['demand']
        estimated_competitor_spending = competitor_price * demand * 0.05  # Assume 5% of revenue
        return {'marketing_budget': estimated_competitor_spending}
    
    @staticmethod
    def objective_task(scenario_data: Dict[str, float], base_amount: float = 50000.0) -> Dict[str, float]:
        """Fixed base amount adjusted for market conditions."""
        market_growth = scenario_data['market_growth']
        growth_factor = 1 + (market_growth * 5)  # Increase marketing in growing markets
        marketing_budget = base_amount * growth_factor
        return {'marketing_budget': marketing_budget}