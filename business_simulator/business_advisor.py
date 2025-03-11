import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple, Any
import pickle

# Import our modules
from business_simulator.business_environment import BusinessEnvironment
from business_simulator.reinforcement_learning_agent import DQNAgent
from business_simulator.train_and_evaluate import create_business_decision_agent

class BusinessAdvisor:
    """
    Business advisor that uses a trained RL agent to provide decision recommendations
    based on real business data.
    """
    
    def __init__(self, model_path, config=None):
        """
        Initialize the business advisor.
        
        Args:
            model_path: Path to the trained model
            config: Configuration for the business environment
        """
        self.model_path = model_path
        self.config = config if config else {}
        
        # Create a reference environment
        self.env = BusinessEnvironment(config=config)
        
        # Load any custom data mappings if available
        self.feature_mapping = {}
        
        try:
            mapping_path = f"{model_path}_feature_mapping.pkl"
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.feature_mapping = pickle.load(f)
        except Exception as e:
            print(f"Feature mapping file not found or error loading: {str(e)}. Using default mappings.")
        
        # Create the agent with error handling    
        self.agent = create_business_decision_agent(model_path, config)
        
    def map_real_data_to_state(self, business_data):
        """
        Map real business data to environment state representation.
        
        Args:
            business_data: Dictionary of real business metrics
            
        Returns:
            state: Environment state vector
        """
        # Default mapping (can be customized for specific business contexts)
        state = np.zeros(self.env.observation_space.shape[0], dtype=np.float32)
        
        # Map cash or equivalent
        state[0] = business_data.get('cash', 0)
        
        # Map inventory
        state[1] = business_data.get('inventory', 0)
        
        # Map demand/sales
        state[2] = business_data.get('demand', business_data.get('sales', 0))
        
        # Map cost
        state[3] = business_data.get('unit_cost', business_data.get('cost_per_unit', 0))
        
        # Map competitor price
        state[4] = business_data.get('competitor_price', 0)
        
        # Map market growth
        state[5] = business_data.get('market_growth', 0)
        
        # Apply any custom feature mappings
        for feature_idx, mapping_func in self.feature_mapping.items():
            state[feature_idx] = mapping_func(business_data)
        
        return state
    
    def get_recommendation(self, business_data):
        """
        Get decision recommendations based on current business data.
        
        Args:
            business_data: Dictionary of business metrics
            
        Returns:
            recommendations: Dictionary of recommended actions and explanations
        """
        # Map business data to state representation
        state = self.map_real_data_to_state(business_data)
        
        # Get action recommendation from agent
        action = self.agent.act(state)
        
        # Extract action components
        price, production_qty, marketing_budget = action
        
        # Calculate expected outcomes (simplified)
        expected_profit = self._estimate_profit(state, action)
        
        # Prepare recommendations
        recommendations = {
            'recommended_actions': {
                'price': float(price),
                'production_quantity': float(production_qty),
                'marketing_budget': float(marketing_budget)
            },
            'expected_outcomes': {
                'estimated_profit': float(expected_profit),
            },
            'explanation': self._generate_explanation(state, action, expected_profit)
        }
        
        return recommendations
    
    def _estimate_profit(self, state, action):
        """
        Estimate profit based on state and action using environment model.
        
        Args:
            state: Business state vector
            action: Action vector
            
        Returns:
            profit: Estimated profit
        """
        # Clone environment to avoid modifying original
        env_copy = BusinessEnvironment(config=self.config)
        
        # Set environment to match state (simplified)
        env_copy.cash = state[0]
        env_copy.inventory = state[1]
        env_copy.current_demand = state[2]
        env_copy.current_cost = state[3]
        env_copy.competitor_price = state[4]
        env_copy.market_growth = state[5]
        
        # Take action and get result
        _, reward, _, _, info = env_copy.step(action)
        
        # Return profit estimation
        return info.get('profit', 0)
    
    def _generate_explanation(self, state, action, expected_profit):
        """
        Generate explanation for the recommendation.
        
        Args:
            state: Business state vector
            action: Action vector
            expected_profit: Expected profit from the action
            
        Returns:
            explanation: Explanation text
        """
        price, production_qty, marketing_budget = action
        cash, inventory, demand, cost, competitor_price, market_growth = state
        
        # Price explanation
        if price > competitor_price * 1.1:
            price_strategy = "premium pricing"
            price_reason = "Your cost structure and market position support charging a premium."
        elif price < competitor_price * 0.9:
            price_strategy = "competitive pricing"
            price_reason = "A competitive price will help drive sales volume in the current market."
        else:
            price_strategy = "balanced pricing"
            price_reason = "A price close to competitors maintains market balance."
        
        # Production explanation
        inventory_months = inventory / demand if demand > 0 else float('inf')
        if production_qty > demand * 0.8:
            production_reason = f"Current inventory of {inventory:.0f} units covers approximately {inventory_months:.1f} months of demand. Increasing production anticipates growth and prevents stockouts."
        elif production_qty < demand * 0.2:
            production_reason = f"With {inventory:.0f} units in inventory ({inventory_months:.1f} months of demand), limited production is recommended to avoid excess inventory costs."
        else:
            production_reason = f"Production balanced with demand to maintain optimal inventory levels."
        
        # Marketing explanation
        if marketing_budget > 0.3 * cash:
            marketing_reason = "Substantial marketing investment recommended to drive growth in the current market conditions."
        elif marketing_budget < 0.05 * cash:
            marketing_reason = "Limited marketing recommended to preserve cash while maintaining minimal market presence."
        else:
            marketing_reason = "Balanced marketing approach to support steady growth without excessive expenditure."
        
        # Combine explanations
        explanation = f"""
Based on your current business metrics and market conditions, I recommend a {price_strategy} strategy with a price of ${price:.2f} per unit. {price_reason}

For production, I recommend producing {production_qty:.0f} units. {production_reason}

Allocate ${marketing_budget:.2f} to marketing efforts. {marketing_reason}

These recommendations are expected to generate approximately ${expected_profit:.2f} in profit based on the current conditions.
        """
        
        return explanation.strip()
    
    def analyze_scenario(self, business_data, scenarios=None):
        """
        Analyze multiple scenarios and provide comparative recommendations.
        
        Args:
            business_data: Base business data
            scenarios: List of scenario modifications
            
        Returns:
            analysis: Dictionary of scenario analyses
        """
        if scenarios is None:
            # Default scenarios
            scenarios = [
                {'name': 'Current', 'changes': {}},
                {'name': 'High Demand', 'changes': {'demand': business_data.get('demand', 0) * 1.2}},
                {'name': 'Low Demand', 'changes': {'demand': business_data.get('demand', 0) * 0.8}},
                {'name': 'Increased Competition', 'changes': {'competitor_price': business_data.get('competitor_price', 0) * 0.9}},
                {'name': 'Rising Costs', 'changes': {'unit_cost': business_data.get('unit_cost', business_data.get('cost_per_unit', 0)) * 1.1}}
            ]
        
        # Analyze each scenario
        results = []
        for scenario in scenarios:
            # Create scenario data
            scenario_data = business_data.copy()
            scenario_data.update(scenario['changes'])
            
            # Get recommendation
            recommendation = self.get_recommendation(scenario_data)
            
            # Store scenario results
            results.append({
                'scenario': scenario['name'],
                'recommendation': recommendation['recommended_actions'],
                'expected_profit': recommendation['expected_outcomes']['estimated_profit']
            })
        
        # Create analysis summary
        analysis = {
            'scenarios': results,
            'visualizations': self._create_scenario_visualizations(results)
        }
        
        return analysis
    
    def _create_scenario_visualizations(self, scenario_results):
        """
        Create visualizations for scenario analysis.
        
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
        
        # Placeholder for visualization data
        # In a real application, you would save the figures or convert to web-friendly format
        visualizations = {
            'profit_comparison': {'scenarios': scenarios, 'values': profits},
            'price_comparison': {'scenarios': scenarios, 'values': prices},
            'production_comparison': {'scenarios': scenarios, 'values': production},
            'marketing_comparison': {'scenarios': scenarios, 'values': marketing}
        }
        
        return visualizations
    
    def train_on_historical_data(self, historical_data, epochs=10):
        """
        Fine-tune the agent using historical business data.
        
        Args:
            historical_data: List of historical business states, actions, and outcomes
            epochs: Number of training epochs
            
        Returns:
            training_metrics: Dictionary of training metrics
        """
        # Track training metrics
        metrics = {
            'losses': []
        }
        
        # Process historical data
        for epoch in range(epochs):
            epoch_losses = []
            
            for data_point in historical_data:
                # Extract historical data
                state_data = data_point['state']
                action_data = data_point['action']
                next_state_data = data_point['next_state']
                reward = data_point['profit']  # Use profit as reward
                done = data_point.get('terminal', False)
                
                # Map to environment state representation
                state = self.map_real_data_to_state(state_data)
                next_state = self.map_real_data_to_state(next_state_data)
                
                # Map to discrete action
                # Simplified mapping - in real implementation, would need inverse of agent's action mapping
                discrete_action = self._map_to_discrete_action(action_data)
                
                # Learn from this experience
                self.agent.remember(state, discrete_action, reward, next_state, done)
                loss = self.agent.replay()
                
                if loss > 0:
                    epoch_losses.append(loss)
            
            # Track average loss for this epoch
            if epoch_losses:
                metrics['losses'].append(np.mean(epoch_losses))
            else:
                metrics['losses'].append(0)
            
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {metrics['losses'][-1]:.4f}")
        
        # Save updated model
        self.agent.save(self.model_path + "_finetuned")
        
        return metrics
    
    def _map_to_discrete_action(self, action_data):
        """
        Map continuous business action to discrete action index.
        Simplified version - real implementation would need to match the agent's action space.
        
        Args:
            action_data: Dictionary with continuous action values
            
        Returns:
            discrete_action: Discrete action index
        """
        # Extract action components
        price = action_data.get('price', 0)
        production_qty = action_data.get('production_quantity', 0)
        marketing_budget = action_data.get('marketing_budget', 0)
        
        # Find closest price level
        price_factor = price / self.env.current_cost if self.env.current_cost > 0 else 1.3
        price_levels = self.agent.price_levels
        price_idx = min(range(len(price_levels)), key=lambda i: abs(price_levels[i] - price_factor))
        
        # Find closest production level
        prod_factor = production_qty / self.env.config['max_production'] if self.env.config['max_production'] > 0 else 0.5
        prod_levels = self.agent.production_levels
        prod_idx = min(range(len(prod_levels)), key=lambda i: abs(prod_levels[i] - prod_factor))
        
        # Find closest marketing level
        mktg_factor = marketing_budget / self.env.config['max_marketing_budget'] if self.env.config['max_marketing_budget'] > 0 else 0.5
        mktg_levels = self.agent.marketing_levels
        mktg_idx = min(range(len(mktg_levels)), key=lambda i: abs(mktg_levels[i] - mktg_factor))
        
        # Generate discrete action index
        for idx, mapping in self.agent.action_mapping.items():
            if mapping == (price_idx, prod_idx, mktg_idx):
                return idx
        
        # Default to middle action if no match found
        return list(self.agent.action_mapping.keys())[len(self.agent.action_mapping) // 2]
    
    def save_feature_mapping(self, feature_mapping):
        """
        Save custom feature mapping for business data.
        
        Args:
            feature_mapping: Dictionary of feature indices to mapping functions
        """
        self.feature_mapping = feature_mapping
        
        # Save mapping to file
        with open(f"{self.model_path}_feature_mapping.pkl", 'wb') as f:
            pickle.dump(feature_mapping, f)