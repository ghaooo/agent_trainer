# Business Advisor and Simulator

A Python package for simulating business operations, training reinforcement learning agents to make optimal business decisions, and providing strategic business recommendations.

## Overview

This project consists of two main components:

1. **Business Simulator**: A simulation environment for business operations that can be used to train reinforcement learning agents.
2. **Business Advisor**: A recommendation system that uses trained models to provide strategic business advice across pricing, production, and marketing decisions.

## Project Structure

```
.
├── business_simulator/     # Core simulation environment
│   ├── business_environment.py # Business simulation environment
│   ├── reinforcement_learning_agent.py # DQN agent implementation
│   ├── business_advisor.py # Business recommendation system
│   ├── agents.py          # Simple agent implementations
│   ├── simulation.py      # Simulation utilities
│   └── train_and_evaluate.py # Training and evaluation utilities
├── business_advisor/      # Business strategy recommendation modules
│   ├── pricing.py         # Pricing strategy implementations
│   ├── marketing.py       # Marketing strategy implementations
│   ├── production.py      # Production strategy implementations
│   ├── evaluation.py      # Strategy evaluation utilities
│   ├── visualization.py   # Data visualization utilities
│   └── web_app.py         # Web interface components
├── playground/            # Demo scripts
│   ├── basic_simulation_demo.py # Demo for basic simulation
│   ├── business_advisor_demo.py # Demo for business advisor
│   ├── price_analysis_demo.py   # Demo for price analysis
│   └── training_demo.py   # Demo for training RL agents
├── models/                # Trained model storage
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/business-advisor.git
cd business-advisor
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running a Basic Simulation

To run a basic business simulation with a heuristic agent:

```bash
python playground/basic_simulation_demo.py
```

### Training a Reinforcement Learning Agent

To train a DQN agent to make optimal business decisions:

```bash
python playground/training_demo.py
```

### Using the Business Advisor

To get business recommendations based on a trained model and explore different strategies:

```bash
python playground/business_advisor_demo.py
```

This will launch a Gradio web interface where you can input business parameters and receive recommendations.

## Business Environment

The business environment simulates a company making decisions about:

- **Pricing**: Setting optimal product prices
- **Production**: Determining production quantities
- **Marketing**: Allocating marketing budget

The environment includes market dynamics such as:

- Price elasticity of demand
- Market growth rates
- Competitor pricing
- Inventory management
- Cash flow constraints

## Business Advisor

The Business Advisor provides strategic recommendations using:

1. **Model-based recommendations**: Using trained reinforcement learning models
2. **Strategy comparison**: Evaluating different pricing strategies:
   - Cost-plus pricing
   - Competitor matching
   - Premium pricing
   - Value-based pricing
   - Penetration pricing
3. **Sensitivity analysis**: Testing how changes in market conditions affect optimal decisions
4. **Custom decision simulation**: Evaluating user-defined business decisions

## Development

### Adding New Strategies

To add new pricing, production, or marketing strategies:

1. Add your strategy function to the appropriate module in the `business_advisor` package
2. Update the strategy dictionary in `evaluation.py` to include your new strategy

### Training Custom Models

To train a model with custom parameters:

1. Modify the environment parameters in `business_environment.py`
2. Adjust the training parameters in `training_demo.py`
3. Run the training script and save your model

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.