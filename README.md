# Evolutionary Legged Robotics

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyBullet](https://img.shields.io/badge/physics-PyBullet-green)](https://pybullet.org)

A Python implementation of evolutionary robotics for legged robots using the PyBullet physics engine. This repository contains code for developing and optimizing adaptive locomotion patterns using evolutionary algorithms and neural networks.

## Overview

This project is a Python port of evolutionary robotics code originally implemented in C++ using the Open Dynamics Engine (ODE). The implementation uses PyBullet for physics simulation and implements various evolutionary algorithms to generate effective locomotion patterns for multi-legged robots.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/evo-legged-robotics.git
cd evo-legged-robotics

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Project Structure

- `src/robot/` - Robot definition and parameters
- `src/evolution/` - Evolutionary algorithm implementations
- `src/controllers/` - Neural network and locomotion controllers
- `src/simulation/` - PyBullet environment and visualization
- `examples/` - Example scripts for various scenarios
- `docs/` - Documentation and resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.
