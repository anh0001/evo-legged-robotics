#!/usr/bin/env python3
"""
Visualize the legged robot URDF to verify it looks correct.
"""

import pybullet as p
import pybullet_data
import time
import os

URDF_PATH = "src/robot/urdf/legged_robot.urdf"

# First, generate the URDF if it doesn't exist
if not os.path.exists(URDF_PATH):
    print("URDF file doesn't exist, generating it...")
    from src.robot.urdf_generator import generate_urdf, save_urdf
    urdf_content = generate_urdf()
    save_urdf(urdf_content)

# Connect to PyBullet in GUI mode
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set up camera
p.resetDebugVisualizerCamera(
    cameraDistance=3.0,
    cameraYaw=101.0,
    cameraPitch=-27.5,
    cameraTargetPosition=[0, 0, 0.5]
)

# Configure visualizer
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

# Set gravity
p.setGravity(0, 0, -9.81)

# Load plane
plane_id = p.loadURDF("plane.urdf")

# Load robot URDF
robot_start_pos = [0, 0, 0.5]
robot_id = p.loadURDF(
    URDF_PATH,
    basePosition=robot_start_pos,
    useFixedBase=False
)

# Print joint information
print(f"Robot has {p.getNumJoints(robot_id)} joints")
for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")

# Enable joint angle sliders for interactive control
joint_sliders = []
for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    
    # Only create sliders for non-fixed joints
    if joint_type != p.JOINT_FIXED:
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        
        # Use default limits if not specified
        if lower_limit == 0 and upper_limit == 0:
            lower_limit = -3.14
            upper_limit = 3.14
        
        slider_id = p.addUserDebugParameter(
            joint_name, 
            lower_limit, 
            upper_limit, 
            0  # initial value
        )
        joint_sliders.append((i, slider_id))

print(f"Created {len(joint_sliders)} joint control sliders")

# Main loop
for _ in range(10000):  # Run for 10000 steps
    # Update joint positions from sliders
    for joint_idx, slider_id in joint_sliders:
        angle = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle,
            force=5.0
        )
    
    # Step simulation
    p.stepSimulation()
    
    # Slow down simulation
    time.sleep(0.01)

# Disconnect from PyBullet
p.disconnect()