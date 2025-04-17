#!/usr/bin/env python3
"""
Demonstrate the legged robot URDF with the locomotion system.
"""

import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import sys

# Ensure path includes our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

URDF_PATH = "src/robot/urdf/legged_robot.urdf"

# First, generate the URDF if it doesn't exist
if not os.path.exists(URDF_PATH):
    print("URDF file doesn't exist, generating it...")
    from src.robot.urdf_generator import generate_urdf, save_urdf
    urdf_content = generate_urdf()
    save_urdf(urdf_content)

# Import our modules
from src.robot.leg_robot import LeggedRobot  # Updated to use URDF
from src.controllers.locomotion import LocomotionGenerator
from src.simulation.environment import Environment

def run_demo(gait_type="tripod"):
    """
    Run a demonstration with the specified gait type.
    
    Args:
        gait_type: Type of gait to use (tripod, wave, ripple, turn_left, turn_right)
    """
    # Create environment
    env = Environment(render=True)
    
    # Create robot (using the URDF)
    robot_start_pos = [0, 0, 0.5]
    robot_id = p.loadURDF(
        URDF_PATH,
        basePosition=robot_start_pos,
        useFixedBase=False
    )
    robot = LeggedRobot(client=env.client, urdf_path=URDF_PATH)
    env.add_robot(robot)
    
    # Create locomotion generator
    locomotion = LocomotionGenerator(robot)
    
    # Select the gait type
    if gait_type == "tripod":
        locomotion.define_tripod_gait()
    elif gait_type == "wave":
        locomotion.define_wave_gait()
    elif gait_type == "ripple":
        locomotion.define_ripple_gait()
    elif gait_type == "turn_left":
        locomotion.define_turn_left_gait()
    elif gait_type == "turn_right":
        locomotion.define_turn_right_gait()
    else:
        print(f"Unknown gait type: {gait_type}, defaulting to tripod")
        locomotion.define_tripod_gait()
    
    print(f"Running demo with {gait_type} gait")
    
    # Run simulation
    for i in range(2000):  # Run for 2000 steps
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
        
        # Sleep to make visualization smoother
        time.sleep(0.01)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Legged Robot URDF Demo")
    parser.add_argument("--gait", type=str, default="tripod",
                        choices=["tripod", "wave", "ripple", "turn_left", "turn_right"],
                        help="Gait type to use for the demo")
    
    args = parser.parse_args()
    run_demo(args.gait)