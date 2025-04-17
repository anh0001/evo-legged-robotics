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
