#!/usr/bin/env python3
"""
Main entry point for the evolutionary legged robotics simulation.
This script demonstrates how to use the various components.
"""

import os
import numpy as np
import argparse
import time
import pickle

import pybullet as p

from src.robot.leg_robot import LeggedRobot
from src.simulation.environment import Environment, TrainingEnvironment
from src.controllers.neural_network import NeuralController, AdaptiveController
from src.controllers.neuro_evolutionary import NeuroEvolutionaryController
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
    # --- handle adaptive mode separately, it does its own setup ---
    if mode == "adaptive":
        run_adaptive_demo(render)
        return

    # Create environment and robot for all *other* modes
    env = Environment(render=render)
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
    # Initialize locomotion generator for fallback gait
    locomotion = LocomotionGenerator(robot)
    locomotion.define_tripod_gait()
    
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
            if avg_stability < 0.8:  # Make this higher so it's more likely to trigger
                use_neural = True
            elif avg_stability > 0.9:
                use_neural = False
        
        if use_neural:
            corner_leg_angles = nn_controller.predict(state)
            target_angles = np.zeros((6, 3))
            corner_legs = [0, 2, 3, 5]
            for idx, leg_idx in enumerate(corner_legs):
                target_angles[leg_idx] = corner_leg_angles[idx*3:(idx+1)*3]
        else:
            target_angles = locomotion.get_next_angles()
        
        robot.set_target_angles(target_angles)
        robot.apply_target_angles()
        
        env.step()
        
        if i % 100 == 0:
            pos = robot.get_position()
            print(f"Execution Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), Neural: {use_neural}")


def run_adaptive_demo(render=True):
    """
    Run a demonstration with adaptive control.
    
    Args:
        render: Whether to render the simulation
    """
    # Adaptive mode: set up its own environment and robot
    env = Environment(render=render, terrain_type="rough")
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


def run_neuro_evolutionary_demo(render=True):
    """
    Run a demonstration with integrated neural-evolutionary control.
    This implements the approach from main04.cpp with:
    - Neural network adaptation based on vertical orientation
    - 5x5 grid of obstacle boxes
    - Integrated learning between evolutionary and neural approaches
    
    Args:
        render: Whether to render the simulation
    """
    # Create environment with obstacles
    env = Environment(render=render, terrain_type="obstacles")
    robot = LeggedRobot(client=env.client)
    env.add_robot(robot)
    
    # Check if we have a pre-trained controller
    model_path = "models/neuro_evolutionary_controller"
    
    if os.path.exists(model_path + ".pkl"):
        print(f"Loading pre-trained neuro-evolutionary controller from {model_path}")
        # Load controller
        ne_controller = NeuroEvolutionaryController.load(model_path)
    else:
        print("No pre-trained controller found. Creating a new one...")
        # Create neural-evolutionary controller with dimensions matching main04.cpp
        ne_controller = NeuroEvolutionaryController(input_dim=15, hidden_dim=20, output_dim=12)
        # Save controller
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ne_controller.save(model_path)
    
    # Initialize tracking variables
    learning_count = 0
    total_reward = 0
    stability_history = []
    
    # Run simulation
    for i in range(1000):
        # Get current state
        state = robot.get_state()
        
        # Get target angles from controller
        target_angles = ne_controller.get_actions(state)
        
        # Apply to robot
        robot.set_target_angles(target_angles)
        robot.apply_target_angles()
        
        # Step simulation
        env.step()
        
        # Learn from experience
        if ne_controller.should_learn():
            loss = ne_controller.learn(state)
            learning_count += 1
            if i % 50 == 0:
                print(f"Learning event #{learning_count}, Loss: {loss:.6f}")
        
        # Track stability (vertical orientation)
        if 'rotation_matrix' in state:
            rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
            stability = rot_matrix[2, 2]  # z-component of z-axis (vertical)
            stability_history.append(stability)
        
        # Calculate reward (simple distance metric)
        if i > 0:
            prev_pos = np.array(state['position'])
            curr_pos = np.array(robot.get_position())
            reward = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_reward += reward
        
        # Print progress every 100 steps
        if i % 100 == 0:
            pos = robot.get_position()
            print(f"Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Using Neural: {ne_controller.use_neural}")
    
    # Save the trained controller
    ne_controller.save(model_path)
    
    # Close environment
    env.close()
    
    # Print summary
    print(f"Demo completed. Learning events: {learning_count}, Total reward: {total_reward:.2f}")
    if stability_history:
        avg_stability = sum(stability_history) / len(stability_history)
        print(f"Average stability (vertical orientation): {avg_stability:.4f}")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evolutionary Legged Robotics Demo")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "evolution", "neural", "adaptive", "neuro_evolutionary"],
                        help="Which mode to run")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering for faster simulation")
    
    args = parser.parse_args()
    
    # Run the demo based on selected mode
    if args.mode == "neuro_evolutionary":
        run_neuro_evolutionary_demo(render=not args.no_render)
    else:
        run_demo(render=not args.no_render, mode=args.mode)


if __name__ == "__main__":
    main()