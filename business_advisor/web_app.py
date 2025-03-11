"""Web Application Module

This module contains functions for creating web-based visualizations and interactive
components for the business advisor demo.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from .pricing import PricingStrategy
from .marketing import MarketingStrategy
from .production import ProductionStrategy
from .evaluation import simulate_custom_decision

def export_results_to_html(results: List[Dict[str, Any]], sensitivity_results: Dict[str, List[Dict[str, float]]], 
                         comparison_charts: Dict[str, str]) -> str:
    """Export results to a standalone HTML dashboard file.
    
    Args:
        results: List of scenario results
        sensitivity_results: Dictionary of sensitivity analysis results
        comparison_charts: Dictionary of chart file paths
        
    Returns:
        html_file: Path to the generated HTML file
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Business Advisor Analysis Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
            .chart-container { margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .strategy-comparison { display: flex; flex-wrap: wrap; }
            .strategy-chart { flex: 1; min-width: 300px; margin: 10px; }
            .tab-container { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
            .tab-button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }
            .tab-button:hover { background-color: #ddd; }
            .tab-button.active { background-color: #ccc; }
            .tab-content { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
        </style>
    </head>
    <body>
        <h1>Business Advisor Analysis Dashboard</h1>
        
        <div class="tab-container">
            <button class="tab-button active" onclick="openTab(event, 'ScenarioResults')">Scenario Results</button>
            <button class="tab-button" onclick="openTab(event, 'StrategyComparison')">Strategy Comparison</button>
            <button class="tab-button" onclick="openTab(event, 'SensitivityAnalysis')">Sensitivity Analysis</button>
        </div>
        
        <div id="ScenarioResults" class="tab-content" style="display: block;">
            <div class="section">
                <h2>Scenario Analysis Results</h2>
    """
    
    # Add scenario results tables
    for scenario_result in results:
        html_content += f"""
                <div class="card">
                    <h3>Scenario: {scenario_result['scenario_name']}</h3>
                    <h4>Input Parameters:</h4>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        for param, value in scenario_result['input_params'].items():
            html_content += f"<tr><td>{param}</td><td>{value}</td></tr>\n"
        
        html_content += """
                    </table>
                    <h4>Strategy Recommendations:</h4>
                    <table>
                        <tr>
                            <th>Strategy</th>
                            <th>Price ($)</th>
                            <th>Production</th>
                            <th>Marketing ($)</th>
                            <th>Expected Profit ($)</th>
                        </tr>
        """
        
        for strategy_name, strategy_result in scenario_result['strategies'].items():
            html_content += f"""
                        <tr>
                            <td>{strategy_name}</td>
                            <td>{strategy_result['price']:.2f}</td>
                            <td>{strategy_result['production_quantity']:.0f}</td>
                            <td>{strategy_result['marketing_budget']:.2f}</td>
                            <td>{strategy_result['expected_profit']:.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div id="StrategyComparison" class="tab-content">
            <div class="section">
                <h2>Strategy Comparison</h2>
                <div class="strategy-comparison">
    """
    
    # Add strategy comparison charts
    for metric, chart_path in comparison_charts.items():
        html_content += f"""
                    <div class="strategy-chart">
                        <img src="{os.path.basename(chart_path)}" alt="{metric} comparison" style="width: 100%;">
                    </div>
        """
    
    html_content += """
                </div>
            </div>
        </div>
        
        <script>
            function openTab(evt, tabName) {
                // Get all tab content and hide them
                var tabcontent = document.getElementsByClassName("tab-content");
                for (var i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                
                // Get all tab buttons and remove "active" class
                var tabbuttons = document.getElementsByClassName("tab-button");
                for (var i = 0; i < tabbuttons.length; i++) {
                    tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
                }
                
                // Show the current tab and add "active" class to the button
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = f"../web_app/business_advisor_dashboard_{timestamp}.html"
    
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    # Copy chart images to the web_app directory
    for chart_path in comparison_charts.values():
        dest_path = os.path.join("../web_app", os.path.basename(chart_path))
        # Copy the file using read/write
        with open(chart_path, 'rb') as src_file:
            with open(dest_path, 'wb') as dest_file:
                dest_file.write(src_file.read())
    
    return html_file

def create_interactive_app(advisor):
    """Create a Gradio interactive web app for exploring business scenarios.
    
    Args:
        advisor: Business advisor instance
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