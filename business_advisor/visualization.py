"""Visualization Utilities Module

This module contains functions for creating visualizations of business strategy evaluations
and sensitivity analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

def create_scenario_visualizations(scenario_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create visualizations for scenario analysis.
    
    Args:
        scenario_results: List of scenario results
        
    Returns:
        visualizations: Dictionary with visualization data
    """
    # Extract data for visualization
    scenarios = [r['scenario'] for r in scenario_results]
    profits = [r['expected_profit'] for r in scenario_results]
    prices = [r['recommendation']['price'] for r in scenario_results]
    production = [r['recommendation']['production_quantity'] for r in scenario_results]
    marketing = [r['recommendation']['marketing_budget'] for r in scenario_results]
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot expected profits
    plt.subplot(2, 2, 1)
    plt.bar(scenarios, profits)
    plt.title('Expected Profit by Scenario')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)
    
    # Plot recommended prices
    plt.subplot(2, 2, 2)
    plt.bar(scenarios, prices)
    plt.title('Recommended Price by Scenario')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    
    # Plot recommended production
    plt.subplot(2, 2, 3)
    plt.bar(scenarios, production)
    plt.title('Recommended Production by Scenario')
    plt.ylabel('Production Quantity')
    plt.xticks(rotation=45)
    
    # Plot recommended marketing
    plt.subplot(2, 2, 4)
    plt.bar(scenarios, marketing)
    plt.title('Recommended Marketing Budget by Scenario')
    plt.ylabel('Marketing Budget')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Return visualization data
    visualizations = {
        'profit_comparison': {'scenarios': scenarios, 'values': profits},
        'price_comparison': {'scenarios': scenarios, 'values': prices},
        'production_comparison': {'scenarios': scenarios, 'values': production},
        'marketing_comparison': {'scenarios': scenarios, 'values': marketing}
    }
    
    return visualizations

def create_sensitivity_plots(sensitivity_results: Dict[str, List[Dict[str, float]]]) -> Dict[str, str]:
    """Create and save sensitivity analysis plots.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        
    Returns:
        plots: Dictionary mapping parameter names to plot file paths
    """
    plots = {}
    
    for param, results in sensitivity_results.items():
        # Extract data
        values = [r['param_value'] for r in results]
        prices = [r['price'] for r in results]
        productions = [r['production'] for r in results]
        profits = [r['profit'] for r in results]
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Price sensitivity
        axs[0].plot(values, prices, 'o-', linewidth=2)
        axs[0].set_title(f'Price Sensitivity to {param}')
        axs[0].set_xlabel(param)
        axs[0].set_ylabel('Recommended Price ($)')
        axs[0].grid(True)
        
        # Production sensitivity
        axs[1].plot(values, productions, 'o-', linewidth=2)
        axs[1].set_title(f'Production Sensitivity to {param}')
        axs[1].set_xlabel(param)
        axs[1].set_ylabel('Recommended Production')
        axs[1].grid(True)
        
        # Profit sensitivity
        axs[2].plot(values, profits, 'o-', linewidth=2)
        axs[2].set_title(f'Profit Sensitivity to {param}')
        axs[2].set_xlabel(param)
        axs[2].set_ylabel('Expected Profit ($)')
        axs[2].grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"../evaluation_results/sensitivity_{param}_{timestamp}.png"
        plt.savefig(plot_path)
        
        plots[param] = plot_path
    
    return plots

def create_strategy_comparison_charts(results: List[Dict[str, Any]], metrics: List[str] = ['price', 'profit']) -> Dict[str, str]:
    """Create charts comparing different strategies across scenarios.
    
    Args:
        results: List of scenario results
        metrics: List of metrics to compare
        
    Returns:
        charts: Dictionary mapping metrics to chart file paths
    """
    charts = {}
    
    for metric in metrics:
        # Prepare data for the chart
        scenarios = []
        strategy_data = {}
        
        for scenario_result in results:
            scenario_name = scenario_result['scenario_name']
            scenarios.append(scenario_name)
            
            for strategy_name, strategy_result in scenario_result['strategies'].items():
                if strategy_name not in strategy_data:
                    strategy_data[strategy_name] = []
                
                if metric == 'price':
                    value = strategy_result['price']
                elif metric == 'production_quantity':
                    value = strategy_result['production_quantity']
                elif metric == 'marketing_budget':
                    value = strategy_result['marketing_budget']
                elif metric == 'profit':
                    value = strategy_result['expected_profit']
                
                strategy_data[strategy_name].append(value)
        
        # Create the chart
        plt.figure(figsize=(12, 6))
        
        # Plot each strategy
        bar_width = 0.15
        index = np.arange(len(scenarios))
        
        for i, (strategy_name, values) in enumerate(strategy_data.items()):
            plt.bar(index + i*bar_width, values, bar_width, label=strategy_name)
        
        plt.xlabel('Scenario')
        plt.ylabel(f'{metric.replace("_", " ").title()}')
        plt.title(f'Strategy Comparison by {metric.replace("_", " ").title()}')
        plt.xticks(index + bar_width * (len(strategy_data) - 1) / 2, scenarios, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = f"../evaluation_results/strategy_comparison_{metric}_{timestamp}.png"
        plt.savefig(chart_path)
        plt.close()
        
        charts[metric] = chart_path
    
    return charts