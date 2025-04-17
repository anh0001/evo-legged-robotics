#!/bin/bash

# Script to create the repository structure for evo-legged-robotics

# Ensure the script stops on errors
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up repository structure for evo-legged-robotics...${NC}"

# Create main directory
REPO_NAME="evo-legged-robotics"
if [ -d "$REPO_NAME" ]; then
    echo -e "${YELLOW}Directory $REPO_NAME already exists. Overwrite? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf "$REPO_NAME"
    else
        echo -e "${RED}Aborting setup.${NC}"
        exit 1
    fi
fi

mkdir -p "$REPO_NAME"
cd "$REPO_NAME"

# Create directory structure
echo -e "${GREEN}Creating directory structure...${NC}"

# src directory and subdirectories
mkdir -p src/robot
mkdir -p src/evolution
mkdir -p src/controllers
mkdir -p src/simulation
mkdir -p src/utils

# examples directory
mkdir -p examples

# docs directory
mkdir -p docs/images

# tests directory
mkdir -p tests

# data and models directories
mkdir -p data
mkdir -p models

# Create __init__.py files for all Python packages
echo -e "${GREEN}Creating Python package files...${NC}"
find src -type d -exec touch {}/__init__.py \;

# Create main source files (empty)
echo -e "${GREEN}Creating source files...${NC}"

# Robot module
touch src/robot/leg_robot.py
touch src/robot/robot_params.py

# Evolution module
touch src/evolution/vega.py
touch src/evolution/ssga.py
touch src/evolution/fitness.py

# Controllers module
touch src/controllers/neural_network.py
touch src/controllers/locomotion.py

# Simulation module
touch src/simulation/environment.py
touch src/simulation/visualization.py

# Utils module
touch src/utils/math_utils.py

# Example files
echo -e "${GREEN}Creating example files...${NC}"
touch examples/basic_robot_sim.py
touch examples/evolution_training.py
touch examples/neural_adaptation.py

# Test files
echo -e "${GREEN}Creating test files...${NC}"
touch tests/test_robot.py
touch tests/test_evolution.py

# Create README files
echo -e "${GREEN}Creating README files...${NC}"

# Main README
cat > README.md << 'EOF'
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
EOF

# Create README for data directory
cat > data/README.md << 'EOF'
# Data Directory

This directory stores experimental data, including:

- Training logs
- Performance metrics
- Evolution statistics
- Fitness history data

Files in this directory are gitignored by default.
EOF

# Create README for models directory
cat > models/README.md << 'EOF'
# Models Directory

This directory stores trained models and controllers, including:

- Neural network weights
- Locomotion sequence parameters
- Optimized controllers

Files in this directory are gitignored by default.
EOF

# Create a basic setup.py
echo -e "${GREEN}Creating setup.py...${NC}"

cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="evo_legged_robotics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pybullet",
        "tensorflow>=2.0.0",
        "matplotlib",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Evolutionary robotics for legged robots using PyBullet",
    keywords="robotics, evolution, pybullet, locomotion",
    url="https://github.com/yourusername/evo-legged-robotics",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
EOF

# Create requirements.txt
echo -e "${GREEN}Creating requirements.txt...${NC}"

cat > requirements.txt << 'EOF'
numpy>=1.19.0
pybullet>=3.2.0
tensorflow>=2.7.0
matplotlib>=3.5.0
pytest>=6.0.0
EOF

# Create a .gitignore file
echo -e "${GREEN}Creating .gitignore...${NC}"

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Data and results
data/*.csv
data/*.json
data/*.txt
data/*.npy
models/*.h5
models/*.pkl

# Logs
logs/
*.log

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# OS specific
.DS_Store
Thumbs.db
EOF

# Copy the ODE documentation to the docs folder
echo -e "${GREEN}Placeholder for ODE documentation...${NC}"
cat > docs/ode_reference.md << 'EOF'
# ODE Reference Documentation

This file is a placeholder for the conversion of the original ODE documentation.
The actual implementation should include information from the ODE documentation PDF.
EOF

# Create system architecture document
cat > docs/system_architecture.md << 'EOF'
# System Architecture

## Overview

This document outlines the architecture of the evo-legged-robotics system, a Python implementation of evolutionary robotics using PyBullet.

## Components

### Robot Model

The robot model consists of:
- Main body
- Articulated legs with configurable degrees of freedom
- Joint control mechanisms

### Evolutionary Algorithms

- Vector Evaluated Genetic Algorithm (VEGA) for multi-objective optimization
- Steady-State Genetic Algorithm (SSGA) for evolving locomotion patterns

### Neural Network Controller

- Adaptive neural controller for rough terrain
- Learning capabilities based on locomotion success

### Simulation Environment

- PyBullet physics simulation
- Various terrain types
- Metrics for fitness evaluation

## Interaction Flow

1. The evolutionary algorithm generates candidate locomotion patterns
2. These patterns are tested in the simulation environment
3. Fitness is evaluated based on performance metrics
4. New generations are created through genetic operators
5. The process repeats until convergence or timeout

## Data Flow

```
Evolutionary Algorithm → Controller Parameters → Robot Control → Simulation → Fitness Evaluation → Evolutionary Algorithm
```

## Extension Points

- Additional terrain types
- Different robot morphologies
- Alternative evolutionary algorithms
- More sophisticated neural controllers
EOF

# Create conversion document
cat > docs/conversion_plan.md << 'EOF'
# Conversion Plan: ODE to PyBullet

This document outlines the strategy for converting the C++ ODE-based robot simulation to Python with PyBullet.

## Mapping ODE Concepts to PyBullet

| ODE Concept | PyBullet Equivalent | Notes |
|-------------|---------------------|-------|
| `dWorldID` | PyBullet client | Use `p.connect(p.GUI)` or `p.connect(p.DIRECT)` |
| `dSpaceID` | Not needed explicitly | PyBullet manages collision detection internally |
| `dBodyID` | Body unique ID | Obtained from `p.loadURDF()` or `p.createMultiBody()` |
| `dJointID` | Constraint ID | Created with `p.createConstraint()` |
| `dGeomID` | Collision shape ID | Part of body creation in PyBullet |
| `dWorldStep()` | `p.stepSimulation()` | Advances the simulation by one step |
| `dJointSetHingeAnchor()` | Part of constraint setup | Specified in `p.createConstraint()` |
| `dJointSetHingeAxis()` | Part of constraint setup | Specified in `p.createConstraint()` |
| `dJointSetHingeParam()` | `p.changeConstraint()` | Controls joint parameters |
| `dBodySetPosition()` | `p.resetBasePositionAndOrientation()` | Sets the position of a body |
| `dBodyGetPosition()` | `p.getBasePositionAndOrientation()` | Gets the position of a body |

## Implementation Strategy

1. Start with a simple robot model to ensure PyBullet integration works correctly
2. Build the environment and basic robot control functions
3. Implement the evolutionary algorithms 
4. Add the neural network controllers
5. Integrate visualization and data logging
EOF

# Add placeholder for example image
echo -e "${GREEN}Creating placeholder image...${NC}"
cat > docs/images/placeholder.txt << 'EOF'
This is a placeholder for the robot simulation image.
In the actual repository, this would be a screenshot or diagram of the robot.
EOF

echo -e "${BLUE}Repository structure has been created successfully!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Add the actual implementation code to the Python files"
echo -e "2. Test the implementation"
echo -e "3. Update documentation with specific details"
echo -e "4. Commit and push to GitHub"

cd ..
echo -e "${GREEN}Done!${NC}"