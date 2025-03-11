# Business Simulator

A Python package for simulating business operations and training reinforcement learning agents to make optimal business decisions.

## Project Structure

```
.
├── business_simulator/     # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── agents.py          # Agent implementations (Random, SimplePolicy)
│   ├── business_advisor.py # Business recommendation system
│   ├── business_environment.py # Business simulation environment
│   ├── reinforcement_learning_agent.py # DQN agent implementation
│   ├── simulation.py      # Simulation utilities
│   └── train_and_evaluate.py # Training and evaluation utilities
├── data/                  # Directory for storing data
├── logs/                  # Directory for storing logs
├── models/                # Directory for storing trained models
├── playground/            # Demo scripts
│   ├── basic_simulation_demo.py # Demo for basic simulation
│   ├── business_advisor_demo.py # Demo for business advisor
│   └── training_demo.py   # Demo for training RL agents
└── README.md              # This file
```

## Installation

Clone the repository and install the required packages:

```bash
git clone <repository-url>
cd business-simulator
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

To get business recommendations based on a trained model:

```bash
python playground/business_advisor_demo.py
```

## Business Environment

The business environment simulates a company making decisions about:

- Pricing
- Production quantities
- Marketing budget

The environment includes market dynamics such as:

- Price elasticity
- Market growth
- Competitor pricing
- Inventory management

## Agents

### Simple Policy Agent

A rule-based agent that follows basic business heuristics.

### Random Agent

An agent that takes random actions, used as a baseline.

### DQN Agent

A Deep Q-Network agent that learns optimal business decisions through reinforcement learning.

## Business Advisor

The BusinessAdvisor class uses a trained reinforcement learning model to provide business recommendations based on current business data. It can also perform scenario analysis to evaluate different business conditions.

## Customization

You can customize the business environment by modifying the configuration parameters in the `BusinessEnvironment` class. See the demo scripts for examples of how to create custom business scenarios.