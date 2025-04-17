# Evolutionary Legged Robotics

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyBullet](https://img.shields.io/badge/physics-PyBullet-green)](https://pybullet.org)

A Python implementation of evolutionary robotics for legged robots using the PyBullet physics engine. This repository contains code for developing and optimizing adaptive locomotion patterns using evolutionary algorithms and neural networks.

![Robot Simulation](https://github.com/yourusername/evo-legged-robotics/raw/main/docs/images/robot_simulation.png)

## Overview

This project is a Python port of evolutionary robotics code originally implemented in C++ using the Open Dynamics Engine (ODE). The implementation uses PyBullet for physics simulation and implements various evolutionary algorithms to generate effective locomotion patterns for multi-legged robots.

The robot model features a main body with multiple articulated legs, each with multiple degrees of freedom. The project includes:

- **Vector Evaluated Genetic Algorithm (VEGA)** for multi-objective optimization
- **Steady-State Genetic Algorithm (SSGA)** for locomotion pattern evolution
- **Neural network controllers** for adaptive locomotion on uneven terrain

## Features

- Multi-legged robot simulation with configurable parameters
- Evolutionary algorithms for motion pattern generation
- Neural network-based adaptive controllers for rough terrain
- Comprehensive fitness evaluation for different locomotion objectives
- Visualization tools for analyzing robot motion and evolution progress

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/evo-legged-robotics.git
cd evo-legged-robotics

# Create a conda environment in the ./env folder with Python 3.7+
conda create --prefix ./evo-legged-env python=3.9 -y
conda activate ./evo-legged-env

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Requirements

- Python 3.7+
- PyBullet
- NumPy
- TensorFlow or PyTorch (for neural network implementation)
- Matplotlib (for visualization)

## Usage Example

The main.py script demonstrates how to use the library:

```python
# Run with standard locomotion
python main.py --mode standard

# Run with evolutionary algorithm
python main.py --mode evolution

# Run with neural network control
python main.py --mode neural

# Run with adaptive control on rough terrain
python main.py --mode adaptive
```

This implementation should provide researchers and developers with a modern, flexible framework for experimenting with evolutionary robotics and legged locomotion.

## Quick Start

```python
from evo_legged_robotics.simulation import Environment
from evo_legged_robotics.robot import LeggedRobot
from evo_legged_robotics.controllers import NeuralController

# Create simulation environment
env = Environment(render=True)

# Initialize robot with default parameters
robot = LeggedRobot()
env.add_robot(robot)

# Load a pre-trained controller
controller = NeuralController.load('models/pretrained_controller.pkl')

# Run simulation
for i in range(1000):
    actions = controller.get_actions(robot.get_state())
    env.step(actions)
    
    if i % 100 == 0:
        print(f"Step {i}: Robot position: {robot.get_position()}")
```

## Evolution Training

Train a new controller using evolutionary algorithms:

```python
from evo_legged_robotics.evolution import VEGA
from evo_legged_robotics.simulation import TrainingEnvironment
from evo_legged_robotics.robot import LeggedRobot

# Create training environment
env = TrainingEnvironment(terrain_type="flat")

# Initialize robot
robot = LeggedRobot()

# Initialize evolutionary algorithm
vega = VEGA(
    population_size=30,
    chromosome_length=10,
    generations=500,
    fitness_objectives=["forward_speed", "energy_efficiency", "stability"]
)

# Train controller (this will take time)
best_controller = vega.train(robot, env, parallel=True)

# Save trained controller
best_controller.save('models/my_controller.pkl')

# Visualize fitness progress
vega.plot_fitness_history()
```

## Project Structure

- `src/robot/` - Robot definition and parameters
- `src/evolution/` - Evolutionary algorithm implementations
- `src/controllers/` - Neural network and locomotion controllers
- `src/simulation/` - PyBullet environment and visualization
- `examples/` - Example scripts for various scenarios
- `docs/` - Documentation and resources

## Background

This project is based on evolutionary robotics research originally developed at Kubota Lab. The original implementation used Open Dynamics Engine (ODE) for physics simulation and implemented genetic algorithms for evolving effective locomotion patterns. This Python implementation aims to modernize the codebase while preserving the core research concepts.

Key concepts from the original implementation:
- Multi-objective fitness evaluation for walking distance, direction control, and energy efficiency
- Evolutionary adaptation to different terrain types
- Neural network controllers that learn from simulated experience

## License

This project is licensed under the MIT License - see the LICENSE file for details.