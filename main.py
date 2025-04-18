#!/usr/bin/env python3
"""
Main entry point for the evolutionary legged robotics simulation.
This script demonstrates how to use the various components.
"""

import argparse
import time
import os
import pickle
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
        robot.apply_target_angles()

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
    """
    # Initialize VEGA with realistic parameters for continuous evolution
    vega = VEGA(
        population_size=30,      # Updated population size to match GAN=30
        chromosome_length=10,    # Updated chromosome length
        generations=100          # Updated minimum generations
    )

    model_path = "models/evolved_controller.pkl"
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained evolutionary controller from {model_path}")
        with open(model_path, 'rb') as f:
            controller = pickle.load(f)
        locomotion = LocomotionGenerator(robot)
        locomotion.set_sequence_controller(controller)
    else:
        print("No pre-trained controller found. Training with continuous evolution...")
        locomotion = LocomotionGenerator(robot)
        # Initial training phase (one evolution run)
        controller = vega.train(robot, TrainingEnvironment(render=False), parallel=False)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(controller, f)
        locomotion.set_sequence_controller(controller)

    iteration = 0
    # Continuous evolution during simulation over 1000 steps
    for i in range(1000):
        if i % 50 == 0:
            iteration += 1
            obj_idx = iteration % 3  # Cycle: 0=forward, 1=left turn, 2=right turn
            vega.generation = iteration
            vega.current_objective = obj_idx  # Use this property in VEGA for multi-objective fitness
            vega.evolve()
            controller = vega.create_controller()
            locomotion.set_sequence_controller(controller)
        angles = locomotion.get_next_angles()
        robot.set_target_angles(angles)
        robot.apply_target_angles()
        env.step()
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
    
    if os.path.exists(model_path + ".weights.h5"):
        print(f"Loading pre-trained neural controller from {model_path}")
        nn_controller = NeuralController.load(model_path)
    else:
        print("No pre-trained controller found. Creating a new one...")
        # Create a neural controller with dimensions matching C++ version
        nn_controller = NeuralController(input_dim=15, hidden_dim=30, output_dim=12)
        
        # Create a locomotion generator for training data
        locomotion = LocomotionGenerator(robot)
        locomotion.define_tripod_gait()
        
        print("Generating training data...")
        
        # Track vertical stability for feedback
        prev_rot_matrix = np.array(robot.get_state()['rotation_matrix']).reshape(3, 3)
        prev_z_dir = prev_rot_matrix[2, 2]
        
        # Collect training data with stability feedback (500 steps)
        for i in range(500):
            all_angles = locomotion.get_next_angles()
            corner_legs = [0, 2, 3, 5]
            target_angles = np.array([all_angles[leg_idx] for leg_idx in corner_legs]).flatten()
            
            # Apply full angles to robot
            robot.set_target_angles(all_angles)
            robot.apply_target_angles()
            
            # Step simulation
            env.step()
            
            # Get current state and compute stability measure
            state = robot.get_state()
            rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
            current_z_dir = rot_matrix[2, 2]
            
            # Learn only when stability (uprightness) improves
            if current_z_dir > prev_z_dir:
                nn_controller.learn(state, target_angles)
                
            prev_z_dir = current_z_dir
            
            if i % 100 == 0:
                pos = robot.get_position()
                print(f"Training Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        print("Performing batch learning...")
        history = nn_controller.batch_learn(epochs=50)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        nn_controller.save(model_path)
    
    # Run simulation with the neural controller
    print("Running with neural controller...")
    
    # Reset robot posture
    robot.reset_posture()
    
    # Execution phase with stability-based switching
    stability_history = []
    use_neural = False
    
    for i in range(1000):
        state = robot.get_state()
        rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
        current_z_dir = rot_matrix[2, 2]
        stability = current_z_dir  # Higher value means more upright
        stability_history.append(stability)
        
        if len(stability_history) > 10:
            avg_stability = np.mean(stability_history[-10:])
            if avg_stability < 0.95:  # Make this higher so it's more likely to trigger
                use_neural = True
            elif avg_stability > 0.98:
                use_neural = False

        use_neural = True  # Force neural controller for demo
        
        if use_neural:
            corner_leg_angles = nn_controller.predict(state)
            target_angles = np.zeros((6, 3))
            corner_legs = [0, 2, 3, 5]
            for idx, leg_idx in enumerate(corner_legs):
                target_angles[leg_idx] = corner_leg_angles[idx*3:(idx+1)*3]
        else:
            locomotion = LocomotionGenerator(robot)
            locomotion.define_tripod_gait()
            target_angles = locomotion.get_next_angles()
        
        robot.set_target_angles(target_angles)
        robot.apply_target_angles()
        
        env.step()
        
        if i % 100 == 0:
            pos = robot.get_position()
            print(f"Execution Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), Neural: {use_neural}")


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
        robot.apply_target_angles()
        
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