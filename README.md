# Evolutionary Legged Robotics

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyBullet](https://img.shields.io/badge/physics-PyBullet-green)](https://pybullet.org)
[![TensorFlow](https://img.shields.io/badge/neural--network-TensorFlow-orange)](https://www.tensorflow.org)

A Python implementation of evolutionary robotics for legged robots using the PyBullet physics engine. This repository contains code for developing and optimizing adaptive locomotion patterns using evolutionary algorithms and neural networks.

![Robot Simulation](https://github.com/anh0001/evo-legged-robotics/raw/main/docs/images/robot_simulation.png)

## Overview

The implementation uses PyBullet for physics simulation and implements various evolutionary algorithms to generate effective locomotion patterns for multi-legged robots.

The robot model features a main body with multiple articulated legs, each with multiple degrees of freedom. The project includes:

- **Virus Evolutionary Genetic Algorithm (VEGA)** for multi-objective optimization
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
- Continuity penalty that discourages abrupt jumps when a gait cycle repeats

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

## Install or Reinstall Mesa DRI Drivers

```bash
sudo apt update
sudo apt install --reinstall libgl1-mesa-dri libglx-mesa0 mesa-utils
```

- `libgl1-mesa-dri` provides the iris/swrast DRI drivers
- `libglx-mesa0` provides the GLX library for Mesa  
- `mesa-utils` gives you tools like `glxinfo` and `glxgears` to verify rendering

After installing, verify direct rendering:

```bash
glxinfo | grep "direct rendering"
```

You should see `Yes`.

### Switch GPU Mode on Hybrid Systems

On laptops with both Intel and NVIDIA GPUs, the wrong profile can prevent loading the Intel (iris) driver. Switch to NVIDIA (or Intel) mode:

```bash
sudo prime-select nvidia
```

or for Intel:

```bash
sudo prime-select intel
sudo reboot
```

## Requirements

- Python 3.7+
- PyBullet
- NumPy
- TensorFlow
- Pandas
- Matplotlib

## Testing

This section provides step-by-step instructions to validate your installation and run quick tests to ensure the framework is working correctly.

### Step 1: Framework Validation

First, validate that all dependencies and core components are properly installed:

```bash
chmod +x scripts/validate_framework.py
```

```bash
python scripts/validate_framework.py
```

This script checks your Python environment, verifies all required packages are installed, and validates the core framework components.

### Step 2: Quick Test Runs

#### Test Enhanced VEGA Algorithm
Run the enhanced Virus Evolutionary Genetic Algorithm (VEGA) with minimal settings to verify the evolutionary core is working:

#### Test with minimal settings
```bash
python experiments/core/run_evolution.py --quick-test --max-iterations=100 --real-time
```

#### Test with full settings
```bash
# For a more comprehensive test with longer iterations
python experiments/core/run_evolution.py --max-iterations=1000 --real-time
```

#### Test Ablation Study Framework
Test the ablation framework with minimal configurations (recommended for first-time users):

```bash
python experiments/studies/ablation_study.py --configs=minimal --runs=3
```

For a comprehensive ablation study (takes longer but tests all configurations):

```bash
python experiments/studies/ablation_study.py --full-study
python experiments/studies/ablation_study.py --config=experiments/configs/ablation_configs.yaml --runs=50
```

#### Test Visualization System
Then test with real experimental data:

```bash
python experiments/visualization/publication_plots_runner.py
```

## Usage Example

```bash
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

# Run with integrated mode (evolution + neural integration)
python main.py --mode integrated
```

### Mode Descriptions

- **standard**: Uses predefined locomotion patterns  
- **evolution**: Uses Virus Evolutionary Genetic Algorithm (VEGA) for locomotion optimization  
- **neural**: Uses neural network for adaptive control  
- **adaptive**: Combines sequence-based locomotion with neural adaptation  
- **neuro_evolutionary**: Combines neural and evolutionary approaches  
- **neuro_adaptive_terrain**: Neural network adaptation based on leg heights for terrain sensing  
- **integrated**: Full integration of evolutionary algorithms and neural adaptation in a single control loop  

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

## Configuration

Two dictionaries control the evaluation process in `VEGA`:

- **fitness_weights** &ndash; multipliers for each normalized fitness objective.
- **penalty_factors** &ndash; coefficients used by multiplicative penalties. All
  factors stay in `[0,1)` so penalties scale fitness smoothly without nullifying
  it.
- **evolution_parameters** &ndash; values such as `crossover_rate` and
  `mutation_prob` that control how aggressively VEGA explores.

## License

This project is licensed under the MIT License - see the LICENSE file for details.