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
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from src.robot.leg_robot import LeggedRobot
from src.simulation.environment import Environment, TrainingEnvironment
from src.controllers.neural_network import NeuralController, AdaptiveController
from src.controllers.neuro_adaptive_terrain import NeuroAdaptiveTerrainController
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
    Run a demonstration with the evolutionary algorithm.
    This version uses the improved VEGA implementation with proper logging.
    
    Args:
        env: Simulation environment
        robot: Robot instance
    
    Returns:
        The evolved controller
    """
    print(f"Running evolution demonstration...")
    
    # Initialize VEGA with parameters matching the C++ implementation
    vega = VEGA(
        population_size=30,      # GAN=30 in C++
        chromosome_length=10     # GAL=10 in C++
    )
    
    # Check if we should load from an existing checkpoint
    model_path = os.path.join('models', 'evolved_controller.pkl')
    if os.path.exists(model_path):
        print(f"Loading pre-trained evolutionary controller from {model_path}")
        controller = VEGA.load_controller(model_path)
        locomotion = LocomotionGenerator(robot)
        locomotion.set_sequence_controller(controller)
    else:
        # Setup parameters for locomotion control from C++ (exact values)
        times = 0
        timesmax = 20     # Steps per sequence position
        vel_counter = 0
        sampling_steps = 20   # Sampling steps for feedback (samstep in C++)
        gain = 5.0        # Control gain for motors
        
        # Store previous state for fitness evaluation
        prev_pos = np.array(robot.get_position())
        prev_state = robot.get_state()
        prev_rot_matrix = np.array(prev_state['rotation_matrix']).reshape(3, 3)
        
        # Set initial angles (matching original C++)
        angles = np.zeros((6, 3))
        for i in range(6):
            for j in range(3):
                if i < 3:  # Right side legs
                    angles[i, j] = -np.radians(vega.q_init[j])
                else:      # Left side legs
                    angles[i, j] = np.radians(vega.q_init[j])
        
        # Set initial angles on robot
        robot.set_target_angles(angles)
        robot.apply_target_angles()
        
        # Maximum iterations for the demo
        max_iterations = 500
        
        # Main simulation loop
        for i in range(max_iterations):
            # Increment time counter
            times += 1
            vel_counter += 1
            
            # Check if it's time to evaluate fitness and update sequence
            if times > timesmax:
                # Get current robot state for fitness calculation
                curr_pos = np.array(robot.get_position())
                curr_state = robot.get_state()
                curr_rot_matrix = np.array(curr_state['rotation_matrix']).reshape(3, 3)
                
                # Calculate fitness
                vega.evaluate_fitness(
                    robot, 
                    prev_pos, curr_pos,
                    prev_rot_matrix, curr_rot_matrix
                )
                
                # Update camera to follow robot
                if env.client == p.GUI:
                    env.update_camera()
                
                # Update stored state for next evaluation
                prev_pos = curr_pos.copy()
                prev_rot_matrix = curr_rot_matrix.copy()
                
                # Move to next iteration
                vega.iteration += 1
                
                # If iterations >= GAN, evolve the population
                if vega.iteration >= vega.gan:
                    vega.evolve()
                    
                # Reset target angles to defaults
                for i in range(6):
                    for j in range(3):
                        if i < 3:  # Right side
                            angles[i, j] = -np.radians(vega.q_init[j])
                        else:      # Left side
                            angles[i, j] = np.radians(vega.q_init[j])
                            
                # Reset sequence position
                vega.gaj = -1
                times = 0
                
                # Save data occasionally
                if vega.iteration % 100 == 0:
                    vega.save_fitness_data()
                    
                # Exit if we've completed enough iterations
                if vega.iteration >= max_iterations // timesmax:
                    break
            else:
                # Update sequence position
                if vega.gaj < 0:
                    vega.gaj = 0
                else:
                    vega.gaj = (vega.gaj + 1) % vega.host_lengths[vega.gai]
                    
                # Get target angles from current sequence position
                angles = vega.get_target_angles()
            
            # Apply angles to robot
            robot.set_target_angles(angles)
            
            # Calculate joint velocities based on current angles vs target angles
            if vel_counter % 2 == 0:
                # Apply velocity control in robot.apply_target_angles()
                if vel_counter % sampling_steps == 0:
                    vel_counter = 0
            
            # Apply target angles to robot
            robot.apply_target_angles()
            
            # Step the simulation
            env.step()
            
            # Print progress
            if i % 100 == 0:
                pos = robot.get_position()
                print(f"Step {i}: Robot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                
                # Optional pause for visualization
                if env.client == p.GUI:
                    time.sleep(0.01)
        
        # Save final data and generate summary
        vega.save_fitness_data()
        vega.plot_fitness_history()
        vega.save_summary()
        
        # Save the best controller for deployment
        controller_path = vega.save_best_controller()
        controller = vega.create_controller()
        
        # Save a copy to the standard location
        with open(model_path, 'wb') as f:
            pickle.dump(controller, f)
    
    print("Evolution completed. Robot now using evolved controller.")
    
    return controller


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


def run_neuro_adaptive_terrain_demo(render=True, log_dir=None):
    """
    Run a demonstration of neural-adaptive terrain locomotion with leg height sensing.
    """
    # allow user to override log directory
    if log_dir is None:
        log_dir = os.path.join('logs', 'neuro_adaptive_terrain', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('neuro_adaptive_terrain')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'demo.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Starting neuro-adaptive terrain locomotion demonstration...")

    env = Environment(render=render, terrain_type="obstacles")
    robot = LeggedRobot(client=env.client)
    env.add_robot(robot)
    model_path = "models/neuro_adaptive_terrain_controller"
    if os.path.exists(model_path + ".pkl"):
        logger.info(f"Loading pre-trained controller from {model_path}")
        controller = NeuroAdaptiveTerrainController.load(model_path)
    else:
        logger.info("No pre-trained controller found. Creating a new one...")
        controller = NeuroAdaptiveTerrainController(input_dim=15, hidden_dim=30, output_dim=12, log_dir=log_dir)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        controller.save(model_path)

    vel_counter = 0
    sampling_steps = 50
    learning_count = 0
    total_reward = 0
    stability_history = []
    total_steps = 2000
    for i in range(total_steps):
        state = get_extended_state(robot)
        vel_counter += 1
        target_angles = controller.get_actions(state)
        robot.set_target_angles(target_angles)
        robot.apply_target_angles()
        env.step()
        if vel_counter % sampling_steps == 0:
            vel_counter = 0
            loss = controller.learn(state)
            if loss > 0:
                learning_count += 1
                if i % 100 == 0:
                    logger.info(f"Step {i}/{total_steps} - Learning event #{learning_count}, Loss: {loss:.6f}")
        if 'rotation_matrix' in state:
            rot = np.array(state['rotation_matrix']).reshape(3, 3)
            stability_history.append(rot[2, 2])
        if i > 0:
            prev = np.array(state['position'])
            curr = np.array(robot.get_position())
            total_reward += np.linalg.norm(curr - prev)
        if i % 100 == 0 or i == total_steps - 1:
            pos = robot.get_position()
            logger.info(f"Step {i}/{total_steps}: Position: ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}), Total Reward: {total_reward:.2f}, Using Neural: {controller.use_neural}")

    controller.save(model_path)
    controller.plot_training_history()
    stability_df = pd.DataFrame({'steps': range(len(stability_history)), 'stability': stability_history})
    stability_df.to_csv(os.path.join(log_dir, 'stability_history.csv'), index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(stability_df['steps'], stability_df['stability'])
    plt.grid(True)
    plt.xlabel('Step')
    plt.ylabel('Vertical Stability')
    plt.title('Robot Stability Throughout Simulation')
    plt.savefig(os.path.join(log_dir, 'stability_history.png'))
    plt.close()
    env.close()
    logger.info(f"Demo completed: Learning events={learning_count}, Total reward={total_reward:.2f}, Avg stability={(sum(stability_history) / len(stability_history)):.4f}")
    logger.info(f"All data saved to: {log_dir}")
    return log_dir


def run_integrated_demo(render=True):
    """
    Run a demonstration using the integrated controller that combines evolutionary 
    algorithms with neural network learning for adaptation to rough terrain.
    This implements the comprehensive approach from main06.cpp.

    Args:
        render: Whether to render the simulation
    """
    from src.controllers.integrated import IntegratedController

    # Create environment with obstacles (like in main06.cpp)
    env = Environment(render=render, terrain_type="obstacles")
    robot = LeggedRobot(client=env.client)
    env.add_robot(robot)
    
    # Create or load integrated controller
    model_path = "models/integrated_controller"
    
    if os.path.exists(model_path + ".params.pkl"):
        print(f"Loading pre-trained integrated controller from {model_path}")
        controller = IntegratedController.load(model_path)
    else:
        print("No pre-trained controller found. Creating a new one...")
        controller = IntegratedController(input_dim=15, hidden_dim=30, output_dim=12)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        controller.save(model_path)
    
    # Setup data collection
    vel_counter = 0
    sampling_steps = 50  # Same as in main06.cpp
    learning_count = 0
    total_reward = 0
    stability_history = []
    total_steps = 2000
    
    # Initial robot position and orientation for calculating fitness
    prev_pos = np.array(robot.get_position())
    prev_orientation = 0
    
    # Run simulation loop
    for i in range(total_steps):
        # Get extended state including leg positions for height sensing
        state = get_extended_state(robot)
        
        # Increment velocity counter (used for learning timing)
        vel_counter += 1
        
        # Get target angles from controller
        target_angles = controller.get_actions(state)
        
        # Apply to robot
        robot.set_target_angles(target_angles)
        robot.apply_target_angles()
        
        # Step simulation
        env.step()
        
        # Learn from experience at regular intervals
        if vel_counter % sampling_steps == 0:
            vel_counter = 0
            
            # Calculate fitness metrics for evolutionary algorithm
            curr_pos = np.array(robot.get_position())
            curr_rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
            
            # Extract current orientation angle (yaw)
            curr_orientation = np.arctan2(curr_rot_matrix[1, 0], curr_rot_matrix[0, 0])
            
            # Calculate displacement and angle change
            displacement = curr_pos - prev_pos
            distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
            
            # Calculate angle change
            angle_change = curr_orientation - prev_orientation
            # Adjust to range [-pi, pi]
            if angle_change > np.pi:
                angle_change -= 2 * np.pi
            elif angle_change < -np.pi:
                angle_change += 2 * np.pi
                
            # Calculate direction alignment
            alignment = 0
            if distance > 0:
                forward_vec = np.array([curr_rot_matrix[0, 0], curr_rot_matrix[1, 0]])
                direction_vec = displacement[:2] / np.linalg.norm(displacement[:2])
                alignment = np.dot(forward_vec, direction_vec)
            
            # Update fitness in controller
            controller.evaluate_fitness(robot, distance, angle_change, alignment)
            
            # Learn from current state
            loss = controller.learn(state)
            if loss > 0:
                learning_count += 1
                if i % 100 == 0:
                    print(f"Step {i}/{total_steps} - Learning event #{learning_count}, Loss: {loss:.6f}")
            
            # Update previous position and orientation for next evaluation
            prev_pos = curr_pos
            prev_orientation = curr_orientation
        
        # Track stability (vertical orientation)
        if 'rotation_matrix' in state:
            rot = np.array(state['rotation_matrix']).reshape(3, 3)
            stability_history.append(rot[2, 2])
        
        # Calculate reward (distance traveled)
        if i > 0:
            prev = np.array(state['position'])
            curr = np.array(robot.get_position())
            reward = np.linalg.norm(curr[:2] - prev[:2])  # xy-plane distance
            total_reward += reward
        
        # Print progress periodically
        if i % 100 == 0 or i == total_steps - 1:
            pos = robot.get_position()
            print(f"Step {i}/{total_steps}: Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                  f"Total Reward: {total_reward:.2f}, Using Neural: {controller.use_neural}")
    
    # Save the trained controller
    controller.save(model_path)
    
    # Generate visualization
    controller.plot_training_history()
    
    # Save stability history
    stability_df = pd.DataFrame({'steps': range(len(stability_history)), 'stability': stability_history})
    stability_df.to_csv(os.path.join(controller.log_dir, 'stability_history.csv'), index=False)
    
    # Plot stability history
    plt.figure(figsize=(10, 6))
    plt.plot(stability_df['steps'], stability_df['stability'])
    plt.grid(True)
    plt.xlabel('Step')
    plt.ylabel('Vertical Stability')
    plt.title('Robot Stability Throughout Simulation')
    plt.savefig(os.path.join(controller.log_dir, 'stability_history.png'))
    plt.close()
    
    # Close environment
    env.close()
    
    # Print summary
    print(f"Demo completed: Learning events={learning_count}, Total reward={total_reward:.2f}, "
          f"Avg stability={(sum(stability_history) / len(stability_history)):.4f}")
    print(f"All data saved to: {controller.log_dir}")
    
    return controller.log_dir


def get_extended_state(robot):
    """Get extended state including leg positions for height sensing."""
    state = robot.get_state()
    leg_positions = []
    for lg in range(robot.leg_count):
        if lg in robot.leg_joints:
            ji = robot.leg_joints[lg][-1]
            link = p.getLinkState(robot.body_id, ji)
            leg_positions.append(link[0])
    state['leg_positions'] = leg_positions
    return state


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evolutionary Legged Robotics Demo")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "evolution", "neural", "adaptive",
                                 "neuro_evolutionary", "neuro_adaptive_terrain", "integrated"],
                        help="Which mode to run")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering for faster simulation")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory to save logs (defaults to logs/MODE/TIMESTAMP)")
    
    args = parser.parse_args()
    
    if args.mode == "integrated":
        run_integrated_demo(render=not args.no_render)
    elif args.mode == "neuro_adaptive_terrain":
        run_neuro_adaptive_terrain_demo(render=not args.no_render, log_dir=args.log_dir)
    elif args.mode == "neuro_evolutionary":
        run_neuro_evolutionary_demo(render=not args.no_render)
    else:
        run_demo(render=not args.no_render, mode=args.mode)


if __name__ == "__main__":
    main()