#!/usr/bin/env python3
"""Enhanced Business Advisor Demo

This script demonstrates how to use the BusinessAdvisor class to get recommendations
based on a trained reinforcement learning model, with detailed evaluation, analysis,
interactive exploration, and pricing strategy comparisons.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import wandb
import json
from datetime import datetime
import gradio as gr
import pandas as pd
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple, Optional

# Add the parent directory to the path so we can import the business_simulator package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from business_simulator.business_environment import BusinessEnvironment
from business_simulator.reinforcement_learning_agent import DQNAgent
from business_simulator.business_advisor import BusinessAdvisor

# Import strategy modules
from business_advisor.pricing import PricingStrategy
from business_advisor.marketing import MarketingStrategy
from business_advisor.production import ProductionStrategy
from business_advisor.evaluation import evaluate_business_strategies, perform_sensitivity_analysis, simulate_custom_decision
from business_advisor.visualization import create_sensitivity_plots, create_strategy_comparison_charts
from business_advisor.web_app import export_results_to_html

# Create output directories
os.makedirs("../models/demo", exist_ok=True)
os.makedirs("../evaluation_results", exist_ok=True)
os.makedirs("../web_app", exist_ok=True)

# Global variable for advisor
g_advisor = None

def demo_business_advisor():
    """Run the business advisor demo with evaluations and visualizations.
    
    Returns:
        advisor: Initialized business advisor
        results: Evaluation results
        app: Gradio app interface
        html_file: Path to exported HTML dashboard
    """
    # Initialize the advisor with a trained model
    advisor = BusinessAdvisor(model_path="dqn_model_final")
    global g_advisor
    g_advisor = advisor
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'Base Case',
            'params': {
                'cash': 1000000,
                'inventory': 5000,
                'demand': 10000,
                'unit_cost': 50,
                'competitor_price': 100,
                'market_growth': 0.05
            }
        },
        {
            'name': 'High Demand',
            'params': {
                'cash': 1000000,
                'inventory': 5000,
                'demand': 15000,
                'unit_cost': 50,
                'competitor_price': 100,
                'market_growth': 0.08
            }
        },
        {
            'name': 'Cost Pressure',
            'params': {
                'cash': 1000000,
                'inventory': 5000,
                'demand': 10000,
                'unit_cost': 65,
                'competitor_price': 100,
                'market_growth': 0.05
            }
        }
    ]
    
    # Run strategy evaluations
    results = evaluate_business_strategies(advisor, scenarios[0], scenarios)
    
    # Perform sensitivity analysis
    sensitivity_params = {
        'demand': np.linspace(5000, 20000, 10),
        'unit_cost': np.linspace(30, 80, 10),
        'competitor_price': np.linspace(70, 130, 10)
    }
    
    sensitivity_results = perform_sensitivity_analysis(
        advisor, scenarios[0]['params'], sensitivity_params)
    
    # Create visualizations
    sensitivity_plots = create_sensitivity_plots(sensitivity_results)
    comparison_charts = create_strategy_comparison_charts(
        results, metrics=['price', 'profit', 'production_quantity', 'marketing_budget'])
    
    # Export results to HTML dashboard
    html_file = export_results_to_html(results, sensitivity_results, comparison_charts)
    
    # Create interactive Gradio app
    app = create_interactive_app(advisor)
    
    return advisor, results, app, html_file

def create_interactive_app(advisor):
    """Create a Gradio interactive web app for exploring business scenarios.
    
    Args:
        advisor: Business advisor instance
        
    Returns:
        demo: Gradio interface
    """
    def predict(cash, inventory, demand, unit_cost, competitor_price, market_growth, 
                price=None, production=None, marketing=None):
        """Run prediction with model or custom values."""
        scenario_data = {
            'cash': cash,
            'inventory': inventory,
            'demand': demand,
            'unit_cost': unit_cost,
            'competitor_price': competitor_price,
            'market_growth': market_growth / 100  # Convert from percentage
        }
        
        # If custom values are provided, use them for simulation
        if price is not None and production is not None and marketing is not None:
            result = simulate_custom_decision(advisor, scenario_data, price, production, marketing)
            
            # Create Pandas DataFrame for strategies comparison
            df = pd.DataFrame([
                {
                    'Strategy': 'Your Decision',
                    'Price': result['custom_actions']['price'],
                    'Production': result['custom_actions']['production_quantity'],
                    'Marketing': result['custom_actions']['marketing_budget'],
                    'Profit': result['profit']
                },
                {
                    'Strategy': 'Model Recommendation',
                    'Price': result['model_recommendation']['recommended_actions']['price'],
                    'Production': result['model_recommendation']['recommended_actions']['production_quantity'],
                    'Marketing': result['model_recommendation']['recommended_actions']['marketing_budget'],
                    'Profit': result['model_recommendation']['expected_outcomes']['estimated_profit']
                }
            ])
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(8, 6))
            
            strategies = ['Your Decision', 'Model Recommendation']
            metrics = ['Price', 'Production/100', 'Marketing/10000', 'Profit/10000']
            
            # Normalize values for better visualization
            data = [
                [price, production/100, marketing/10000, result['profit']/10000],
                [result['model_recommendation']['recommended_actions']['price'], 
                 result['model_recommendation']['recommended_actions']['production_quantity']/100,
                 result['model_recommendation']['recommended_actions']['marketing_budget']/10000,
                 result['model_recommendation']['expected_outcomes']['estimated_profit']/10000]
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, data[0], width, label=strategies[0])
            ax.bar(x + width/2, data[1], width, label=strategies[1])
            
            ax.set_title('Decision Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            
            plt.tight_layout()
            
            return df, fig
        
        # Otherwise, use the advisor to get a recommendation
        else:
            recommendation = advisor.get_recommendation(scenario_data)
            
            # Also run alternative strategies
            strategies = {
                'Model': None,  # Already calculated
                'Cost Plus': PricingStrategy.cost_plus_pricing(scenario_data),
                'Competitor Match': PricingStrategy.competitor_match(scenario_data),
                'Premium': PricingStrategy.premium_pricing(scenario_data),
                'Value Based': PricingStrategy.value_based_pricing(scenario_data),
                'Penetration': PricingStrategy.penetration_pricing(scenario_data)
            }
            
            # Calculate profits for each strategy
            strategy_results = []
            for strategy_name, price_action in strategies.items():
                if strategy_name == 'Model':
                    strategy_results.append({
                        'Strategy': 'Model',
                        'Price': recommendation['recommended_actions']['price'],
                        'Production': recommendation['recommended_actions']['production_quantity'],
                        'Marketing': recommendation['recommended_actions']['marketing_budget'],
                        'Profit': recommendation['expected_outcomes']['estimated_profit']
                    })
                else:
                    price = price_action['price']
                    
                    # Use demand matching for production
                    production = ProductionStrategy.demand_matching(scenario_data)['production_quantity']
                    
                    # Use percentage of revenue for marketing
                    marketing = MarketingStrategy.percentage_of_revenue(
                        scenario_data, price, production)['marketing_budget']
                    
                    # Estimate profit
                    expected_sales = min(production, scenario_data['demand'])
                    revenue = price * expected_sales
                    cost = scenario_data['unit_cost'] * production + marketing
                    profit = revenue - cost
                    
                    strategy_results.append({
                        'Strategy': strategy_name,
                        'Price': price,
                        'Production': production,
                        'Marketing': marketing,
                        'Profit': profit
                    })
            
            # Create DataFrame
            df = pd.DataFrame(strategy_results)
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(df))
            width = 0.2
            
            # Plot normalized metrics
            ax.bar(x - width*1.5, df['Price'], width, label='Price')
            ax.bar(x - width/2, df['Production']/100, width, label='Production/100')
            ax.bar(x + width/2, df['Marketing']/10000, width, label='Marketing/10000')
            ax.bar(x + width*1.5, df['Profit']/10000, width, label='Profit/10000')
            
            ax.set_title('Strategy Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(df['Strategy'], rotation=45)
            ax.legend()
            
            plt.tight_layout()
            
            return df, fig
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Business Decision Advisor")
        
        with gr.Row():
            with gr.Column():
                cash = gr.Number(label="Cash", value=1000000)
                inventory = gr.Number(label="Inventory", value=5000)
                demand = gr.Number(label="Demand", value=10000)
            with gr.Column():
                unit_cost = gr.Number(label="Unit Cost", value=50)
                competitor_price = gr.Number(label="Competitor Price", value=100)
                market_growth = gr.Number(label="Market Growth (%)", value=5)
        
        gr.Markdown("## Optional: Test Your Own Decision")
        with gr.Row():
            price = gr.Number(label="Your Price", value=None)
            production = gr.Number(label="Your Production Quantity", value=None)
            marketing = gr.Number(label="Your Marketing Budget", value=None)
        
        btn = gr.Button("Get Recommendations")
        
        with gr.Row():
            output_table = gr.DataFrame(label="Strategy Comparison")
            output_plot = gr.Plot(label="Visualization")
        
        btn.click(
            fn=predict,
            inputs=[cash, inventory, demand, unit_cost, competitor_price, market_growth,
                   price, production, marketing],
            outputs=[output_table, output_plot]
        )
    
    return demo

if __name__ == "__main__":
    # Run the demo
    advisor, results, app, html_file = demo_business_advisor()
    
    # Launch the Gradio interface
    app.launch(share=True)
    
    print(f"\nResults exported to: {html_file}")