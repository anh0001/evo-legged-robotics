# Evolutionary Legged Robotics

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyBullet](https://img.shields.io/badge/physics-PyBullet-green)](https://pybullet.org)
[![TensorFlow](https://img.shields.io/badge/neural--network-TensorFlow-orange)](https://www.tensorflow.org)

A Python implementation of evolutionary robotics for legged robots using the PyBullet physics engine. This repository contains code for developing and optimizing adaptive locomotion patterns using evolutionary algorithms and neural networks.

![Robot Simulation](https://github.com/yourusername/evo-legged-robotics/raw/main/docs/images/robot_simulation.png)

## Overview

The implementation uses PyBullet for physics simulation and implements various evolutionary algorithms to generate effective locomotion patterns for multi-legged robots.

The robot model features a main body with multiple articulated legs, each with multiple degrees of freedom. The project includes:

- **Vector Evaluated Genetic Algorithm (VEGA)** for multi-objective optimization
- **Steady-State Genetic Algorithm (SSGA)** for locomotion pattern evolution
- **Neural network controllers** for adaptive locomotion on uneven terrain
- **Leg height sensing** for terrain adaptation

## Features

- Multi-legged robot simulation with configurable parameters
- Evolutionary algorithms for motion pattern generation
- Neural network-based adaptive controllers for rough terrain
- Comprehensive fitness evaluation for different locomotion objectives
- Visualization tools for analyzing robot motion and evolution progress
- TensorBoard integration for neural network training analysis
- Comprehensive logging and data collection

## Installation

```bash
# Clone the repository
git clone https://github.com/anh0001/evo-legged-robotics.git
cd evo-legged-robotics

# Create a conda environment in the ./env folder with Python 3.7+
conda create --prefix ./evo-legged-env python=3.9 -y
conda activate ./evo-legged-env
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Requirements

- Python 3.7+
- PyBullet
- NumPy
- TensorFlow
- Pandas
- Matplotlib

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

# Run with neuro-evolutionary approach
python main.py --mode neuro_evolutionary

# Run with neural-adaptive terrain approach
python main.py --mode neuro_adaptive_terrain
```

### Mode Descriptions

- **standard**: Uses predefined locomotion patterns  
- **evolution**: Uses Vector Evaluated Genetic Algorithm (VEGA) for locomotion optimization  
- **neural**: Uses neural network for adaptive control  
- **adaptive**: Combines sequence-based locomotion with neural adaptation  
- **neuro_evolutionary**: Combines neural and evolutionary approaches  
- **neuro_adaptive_terrain**: Neural network adaptation based on leg heights for terrain sensing  

## Neuro-Adaptive Terrain Mode

The `neuro_adaptive_terrain` mode implements a sophisticated neural network-based approach for legged robot locomotion that can adapt to different terrains using leg height sensing. Key features include:

- Two-layer neural network with sigmoid activation for joint angle control  
- Leg height sensing to detect and adapt to uneven terrain  
- Incremental learning based on vertical orientation improvement  
- Dataset management for storing and replacing training examples  
- TensorBoard integration for visualizing training progress  
- Comprehensive data logging using Pandas DataFrames and CSV files  

### Logging and Visualization

Training data and visualizations are stored in the `logs/neuro_adaptive_terrain` directory:

- **TensorBoard logs**: View with `tensorboard --logdir=logs/neuro_adaptive_terrain`  
- **CSV files**: Training data saved for external analysis  
- **PNG plots**: Generated visualizations of training progress  
- **Log files**: Detailed execution logs  

## Quick Start

```python
from evo_legged_robotics.simulation import Environment
from evo_legged_robotics.robot import LeggedRobot
from evo_legged_robotics.controllers import NeuroAdaptiveTerrainController

# Create simulation environment
env = Environment(render=True, terrain_type="obstacles")

# Initialize robot with default parameters
robot = LeggedRobot()
env.add_robot(robot)

# Create controller
controller = NeuroAdaptiveTerrainController()

# Run simulation
for i in range(1000):
    # Get current state with leg positions
    state = get_extended_state(robot)
    
    # Get actions from controller
    actions = controller.get_actions(state)
    
    # Apply to robot and step simulation
    robot.set_target_angles(actions)
    robot.apply_target_angles()
    env.step()
    
    # Learn from experience when orientation improves
    controller.learn(state)
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
- `logs/` - Training logs, visualizations, and data

## Background

This project is based on evolutionary robotics research originally developed at Kubota Lab. The original implementation used Open Dynamics Engine (ODE) for physics simulation and implemented genetic algorithms for evolving effective locomotion patterns. This Python implementation aims to modernize the codebase while preserving the core research concepts.

Key concepts from the original implementation:
- Multi-objective fitness evaluation for walking distance, direction control, and energy efficiency
- Evolutionary adaptation to different terrain types
- Neural network controllers that learn from simulated experience

## License

This project is licensed under the MIT License - see the LICENSE file for details.