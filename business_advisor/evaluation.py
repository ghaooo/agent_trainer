"""Evaluation Utilities Module

This module contains functions for evaluating and comparing different business strategies,
performing sensitivity analysis, and generating comparative visualizations.
"""

from typing import Dict, List, Any
import wandb

from .pricing import PricingStrategy
from .marketing import MarketingStrategy
from .production import ProductionStrategy


def evaluate_business_strategies(advisor, base_scenario: Dict[str, float], variations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate business advisor and alternative strategies across multiple scenarios.
    
    Args:
        advisor: Business advisor instance
        base_scenario: Base business scenario data
        variations: List of scenario variations to evaluate
        
    Returns:
        List of evaluation results for each scenario
    """
    results = []
    strategies = {
        'Model': None,  # Use advisor's recommendation
        'Cost Plus': PricingStrategy.cost_plus_pricing,
        'Competitor Match': PricingStrategy.competitor_match,
        'Premium': PricingStrategy.premium_pricing,
        'Value Based': PricingStrategy.value_based_pricing,
        'Penetration': PricingStrategy.penetration_pricing
    }
    
    for variation in variations:
        # Create scenario data by modifying base scenario
        scenario_data = base_scenario.copy()
        for param, value in variation['params'].items():
            scenario_data[param] = value
        
        scenario_results = {
            'scenario_name': variation['name'],
            'input_params': scenario_data,
            'strategies': {}
        }
        
        # For each strategy...
        for strategy_name, pricing_func in strategies.items():
            # For model-based strategy, use advisor
            if strategy_name == 'Model':
                recommendation = advisor.get_recommendation(scenario_data)
                strategy_result = {
                    'price': recommendation['recommended_actions']['price'],
                    'production_quantity': recommendation['recommended_actions']['production_quantity'],
                    'marketing_budget': recommendation['recommended_actions']['marketing_budget'],
                    'expected_profit': recommendation['expected_outcomes']['estimated_profit'],
                    'explanation': recommendation['explanation']
                }
            # For other strategies, combine pricing with default production/marketing
            else:
                # Get price from strategy
                price_action = pricing_func(scenario_data)
                price = price_action['price']
                
                # Use demand matching for production
                production_action = ProductionStrategy.demand_matching(scenario_data)
                production = production_action['production_quantity']
                
                # Use percentage of revenue for marketing
                marketing_action = MarketingStrategy.percentage_of_revenue(scenario_data, price, production)
                marketing_budget = marketing_action['marketing_budget']
                
                # Estimate profit using simplified model
                unit_cost = scenario_data['unit_cost']
                expected_sales = min(production, scenario_data['demand'])
                revenue = price * expected_sales
                cost = unit_cost * production + marketing_budget
                profit = revenue - cost
                
                explanation = f"Used {strategy_name} pricing with standard production and marketing."
                
                strategy_result = {
                    'price': price,
                    'production_quantity': production,
                    'marketing_budget': marketing_budget,
                    'expected_profit': profit,
                    'explanation': explanation
                }
            
            # Add strategy result
            scenario_results['strategies'][strategy_name] = strategy_result
            
            # Log to wandb
            wandb.log({
                f"scenario_{variation['name']}/strategy_{strategy_name}/price": strategy_result['price'],
                f"scenario_{variation['name']}/strategy_{strategy_name}/production": strategy_result['production_quantity'],
                f"scenario_{variation['name']}/strategy_{strategy_name}/marketing": strategy_result['marketing_budget'],
                f"scenario_{variation['name']}/strategy_{strategy_name}/profit": strategy_result['expected_profit']
            })
        
        results.append(scenario_results)
    
    return results


def perform_sensitivity_analysis(advisor, base_scenario: Dict[str, float], param_ranges: Dict[str, List[float]], 
                               wandb_log: bool = True) -> Dict[str, List[Dict[str, float]]]:
    """Perform sensitivity analysis on key parameters."""
    sensitivity_results = {}
    
    for param, values in param_ranges.items():
        param_results = []
        
        for value in values:
            # Create modified scenario
            test_scenario = base_scenario.copy()
            test_scenario[param] = value
            
            # Get recommendation
            recommendation = advisor.get_recommendation(test_scenario)
            
            # Store result
            result = {
                'param_value': value,
                'price': recommendation['recommended_actions']['price'],
                'production': recommendation['recommended_actions']['production_quantity'],
                'marketing': recommendation['recommended_actions']['marketing_budget'],
                'profit': recommendation['expected_outcomes']['estimated_profit']
            }
            param_results.append(result)
            
            # Log to wandb
            if wandb_log:
                wandb.log({
                    f"sensitivity/{param}/value": value,
                    f"sensitivity/{param}/price": result['price'],
                    f"sensitivity/{param}/production": result['production'],
                    f"sensitivity/{param}/marketing": result['marketing'],
                    f"sensitivity/{param}/profit": result['profit']
                })
        
        sensitivity_results[param] = param_results
    
    return sensitivity_results


def simulate_custom_decision(advisor, scenario_data: Dict[str, float], custom_price: float, 
                           custom_production: float, custom_marketing: float) -> Dict[str, Any]:
    """Simulate profit using model's estimate but with custom decision values."""
    # Get the model's recommendation for comparison
    model_recommendation = advisor.get_recommendation(scenario_data)
    
    # Calculate profit using simplified model
    unit_cost = scenario_data['unit_cost']
    demand = scenario_data['demand']
    
    # Apply price elasticity (simplified)
    price_elasticity = -1.2  # Assume price elasticity of -1.2
    price_ratio = custom_price / model_recommendation['recommended_actions']['price']
    demand_effect = (price_ratio ** price_elasticity)
    adjusted_demand = demand * demand_effect
    
    # Apply marketing effect (simplified)
    marketing_elasticity = 0.3  # Assume marketing elasticity of 0.3
    marketing_ratio = custom_marketing / model_recommendation['recommended_actions']['marketing_budget']
    marketing_effect = (marketing_ratio ** marketing_elasticity)
    final_demand = adjusted_demand * marketing_effect
    
    # Calculate sales and profit
    expected_sales = min(custom_production, final_demand)
    revenue = custom_price * expected_sales
    cost = unit_cost * custom_production + custom_marketing
    profit = revenue - cost
    
    return {
        'custom_actions': {
            'price': custom_price,
            'production_quantity': custom_production,
            'marketing_budget': custom_marketing
        },
        'expected_demand': final_demand,
        'expected_sales': expected_sales,
        'revenue': revenue,
        'costs': cost,
        'profit': profit,
        'model_recommendation': model_recommendation
    }