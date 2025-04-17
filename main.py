#!/usr/bin/env python3
"""
Main entry point for the evolutionary legged robotics simulation.
This script demonstrates how to use the various components.
"""

import argparse
import time
import os
import numpy as np

import pybullet as p

from src.robot.leg_robot import LeggedRobot
from src.simulation.environment import Environment, TrainingEnvironment
from src.controllers.neural_network import NeuralController, AdaptiveController
from src.evolution.vega import VEGA
from src.controllers.locomotion import LocomotionGenerator


def run_demo(render=True, mode="standard"):
    """
    Run a demonstration of the robot in the environment.
    
    Args:
        render: Whether to render the simulation
        mode: Which mode to run ("standard", "evolution", "neural", "adaptive")
    """
    print(f"Running {mode} demonstration...")
    
    # Create environment
    env = Environment(render=render)
    
    # Create robot
    robot = LeggedRobot(client=env.client)
    env.add_robot(robot)
    
    if mode == "standard":
        # Run with manually defined locomotion
        run_standard_demo(env, robot)
    
    elif mode == "evolution":
        # Run with evolutionary algorithm
        run_evolution_demo(env, robot)
    
    elif mode == "neural":
        # Run with neural network control
        run_neural_demo(env, robot)
    
    elif mode == "adaptive":
        # Run with adaptive control
        run_adaptive_demo(env, robot)
    
    else:
        print(f"Unknown mode: {mode}")
    
    # Close environment
    env.close()


def run_standard_demo(env, robot):
    """
    Run a demonstration with standard locomotion patterns.
    
    Args:
        env: Simulation environment
        robot: Robot instance
    """
    # Create locomotion generator
    locomotion = LocomotionGenerator(robot)
    
    # Define a simple walking pattern
    # This creates a simple tripod gait (alternating legs)
    locomotion.define_tripod_gait()
    
    # Run simulation for a fixed number of steps
    for i in range(1000):
        # Get next locomotion step
        angles = locomotion.get_next_angles()
        
        # Apply to robot
        robot.set_target_angles(angles)
        
        # Step simulation
        env.step()
        
        # Print progress every 100 steps
        if i % 100 == 0:
            pos = robot.get_position()
            print(f"Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            # Optional: sleep to make visualization clearer
            if env.client == p.GUI:
                time.sleep(0.01)


def run_evolution_demo(env, robot):
    """
    Run a demonstration with evolutionary algorithm.
    
    Args:
        env: Simulation environment
        robot: Robot instance
    """
    # Check if we have a pre-trained controller
    model_path = "models/evolved_controller.pkl"
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained evolutionary controller from {model_path}")
        # Load controller
        import pickle
        with open(model_path, 'rb') as f:
            controller = pickle.load(f)
    else:
        print("No pre-trained controller found. Training for a short period...")
        # Create a VEGA instance with small population and generations for demo
        vega = VEGA(
            population_size=5,  # Small population for demo
            chromosome_length=3,
            generations=3  # Just a few generations for demo
        )
        
        # Create training environment
        train_env = TrainingEnvironment(render=False)
        
        # Train for a few generations
        controller = vega.train(robot, train_env, parallel=False)
        
        # Save controller
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(controller, f)
    
    # Create a locomotion generator using the evolved controller
    locomotion = LocomotionGenerator(robot)
    locomotion.set_sequence_controller(controller)
    
    # Run simulation with the evolved controller
    for i in range(500):
        # Get next locomotion step
        angles = locomotion.get_next_angles()
        
        # Apply to robot
        robot.set_target_angles(angles)
        
        # Step simulation
        env.step()
        
        # Print progress every 100 steps
        if i % 100 == 0:
            pos = robot.get_position()
            print(f"Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")


def run_neural_demo(env, robot):
    """
    Run a demonstration with neural network control.
    
    Args:
        env: Simulation environment
        robot: Robot instance
    """
    # Check if we have a pre-trained neural controller
    model_path = "models/neural_controller"
    
    if os.path.exists(model_path + ".pkl"):
        print(f"Loading pre-trained neural controller from {model_path}")
        # Load controller
        nn_controller = NeuralController.load(model_path)
    else:
        print("No pre-trained controller found. Creating a new one...")
        # Create a neural controller
        nn_controller = NeuralController()
        
        # Create some training data using a simple locomotion pattern
        locomotion = LocomotionGenerator(robot)
        locomotion.define_tripod_gait()
        
        print("Generating training data...")
        # Collect training data
        for i in range(100):
            # Get target angles from locomotion generator
            target_angles = locomotion.get_next_angles().flatten()
            
            # Get current state
            state = robot.get_state()
            
            # Train neural network
            nn_controller.learn(state, target_angles)
            
            # Apply angles to robot
            robot.set_target_angles(locomotion.get_next_angles())
            
            # Step simulation
            env.step()
        
        # Save controller
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        nn_controller.save(model_path)
    
    # Run simulation with the neural controller
    for i in range(500):
        # Get current state
        state = robot.get_state()
        
        # Get target angles from neural network
        target_angles = nn_controller.predict(state)
        
        # Reshape to match robot's expected format
        target_angles = target_angles.reshape(6, 3)
        
        # Apply to robot
        robot.set_target_angles(target_angles)
        
        # Step simulation
        env.step()
        
        # Print progress every 100 steps
        if i % 100 == 0:
            pos = robot.get_position()
            print(f"Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")


def run_adaptive_demo(env, robot):
    """
    Run a demonstration with adaptive control.
    
    Args:
        env: Simulation environment
        robot: Robot instance
    """
    # Create a more challenging environment with obstacles
    env.close()
    env = Environment(render=True, terrain_type="rough")
    robot = LeggedRobot(client=env.client)
    env.add_robot(robot)
    
    # Check if we have a pre-trained adaptive controller
    model_path = "models/adaptive_controller"
    
    if os.path.exists(model_path + "_params.pkl"):
        print(f"Loading pre-trained adaptive controller from {model_path}")
        # Load controller
        adaptive_controller = AdaptiveController.load(model_path)
    else:
        print("No pre-trained controller found. Creating one with default sequence...")
        
        # First get a simple sequence controller
        locomotion = LocomotionGenerator(robot)
        locomotion.define_tripod_gait()
        sequence_controller = {
            'type': 'sequence_controller',
            'sequence_length': locomotion.num_phases,
            'sequences': locomotion.phase_angles.copy()
        }
        
        # Create adaptive controller with this sequence
        adaptive_controller = AdaptiveController(sequence_controller=sequence_controller)
        
        # Save controller
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        adaptive_controller.save(model_path)
    
    # Run simulation with the adaptive controller
    total_reward = 0
    
    for i in range(1000):
        # Get current state
        state = robot.get_state()
        
        # Get target angles from adaptive controller
        target_angles = adaptive_controller.get_actions(state)
        
        # Reshape to match robot's expected format
        target_angles = target_angles.reshape(6, 3)
        
        # Apply to robot
        robot.set_target_angles(target_angles)
        
        # Step simulation
        env.step()
        
        # Calculate reward (simple distance metric)
        if i > 0:
            prev_pos = state['position']
            curr_pos = robot.get_position()
            reward = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_reward += reward
            
            # Adapt controller
            adaptive_controller.adapt_to_terrain(state, reward)
        
        # Print progress every 100 steps
        if i % 100 == 0:
            pos = robot.get_position()
            print(f"Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), Reward: {total_reward:.2f}")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evolutionary Legged Robotics Demo")
    parser.add_argument("--mode", type=str, default="standard",
                      choices=["standard", "evolution", "neural", "adaptive"],
                      help="Which mode to run")
    parser.add_argument("--no-render", action="store_true",
                      help="Disable rendering for faster simulation")
    
    args = parser.parse_args()
    
    # Run the demo
    run_demo(render=not args.no_render, mode=args.mode)


if __name__ == "__main__":
    main()