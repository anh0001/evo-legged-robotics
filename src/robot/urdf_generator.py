#!/usr/bin/env python3
"""
Generate URDF file for the legged robot based on parameters from the C++ code.
"""

import os
import math

def generate_urdf():
    # Robot parameters from the C++ code
    box_pos = [0.0, 0.0, 0.5]
    box_length = 1.0
    box_width = 0.4
    box_height = 0.2
    box_mass = 1.0
    
    bar_length = 0.1
    bar_width = 0.2
    bar_height = 0.1
    bar_mass = 0.05
    bar_rest = 0.04
    
    # Start URDF content
    urdf_content = """<?xml version="1.0"?>
<robot name="legged_robot">
    <!-- Colors -->
    <material name="blue">
        <color rgba="0.5 0.5 1.0 1.0"/>
    </material>
    <material name="light_blue">
        <color rgba="0.6 0.6 1.0 1.0"/>
    </material>
    <material name="pink">
        <color rgba="1.0 0.0 1.0 1.0"/>
    </material>
    
    <!-- Main body -->
    <link name="base_link">
        <visual>
            <geometry>
                <box size="{0} {1} {2}"/>
            </geometry>
            <material name="blue"/>
        </visual>
        <collision>
            <geometry>
                <box size="{0} {1} {2}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="{3}"/>
            <inertia ixx="{4}" ixy="0.0" ixz="0.0" iyy="{5}" iyz="0.0" izz="{6}"/>
        </inertial>
    </link>
""".format(
        box_length, box_width, box_height, 
        box_mass,
        box_mass * (box_width**2 + box_height**2) / 12.0,  # ixx
        box_mass * (box_length**2 + box_height**2) / 12.0,  # iyy
        box_mass * (box_length**2 + box_width**2) / 12.0    # izz
    )
    
    # Calculate leg positions
    leg_positions = []
    
    # Front right legs
    leg_positions.append([(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width), box_pos[2]])
    leg_positions.append([(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*2, box_pos[2]])
    leg_positions.append([(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*3, box_pos[2]])
    
    # Middle right legs
    leg_positions.append([0.0, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width), box_pos[2]])
    leg_positions.append([0.0, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*2, box_pos[2]])
    leg_positions.append([0.0, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*3, box_pos[2]])
    
    # Back right legs
    leg_positions.append([-(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width), box_pos[2]])
    leg_positions.append([-(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*2, box_pos[2]])
    leg_positions.append([-(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*3, box_pos[2]])
    
    # Front left legs
    leg_positions.append([(box_length-bar_length)*0.5, box_width*0.5+bar_width*0.5+(bar_rest+bar_width), box_pos[2]])
    leg_positions.append([(box_length-bar_length)*0.5, box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*2, box_pos[2]])
    leg_positions.append([(box_length-bar_length)*0.5, box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*3, box_pos[2]])
    
    # Middle left legs
    leg_positions.append([0.0, box_width*0.5+bar_width*0.5+(bar_rest+bar_width), box_pos[2]])
    leg_positions.append([0.0, box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*2, box_pos[2]])
    leg_positions.append([0.0, box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*3, box_pos[2]])
    
    # Back left legs
    leg_positions.append([-(box_length-bar_length)*0.5, box_width*0.5+bar_width*0.5+(bar_rest+bar_width), box_pos[2]])
    leg_positions.append([-(box_length-bar_length)*0.5, box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*2, box_pos[2]])
    leg_positions.append([-(box_length-bar_length)*0.5, box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*3, box_pos[2]])
    
    # Calculate dummy leg positions (attachment points)
    dummy_positions = []
    
    # Right side dummy legs
    dummy_positions.append([(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5, box_pos[2]])
    dummy_positions.append([0.0, -box_width*0.5-bar_width*0.5, box_pos[2]])
    dummy_positions.append([-(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5, box_pos[2]])
    
    # Left side dummy legs
    dummy_positions.append([(box_length-bar_length)*0.5, box_width*0.5+bar_width*0.5, box_pos[2]])
    dummy_positions.append([0.0, box_width*0.5+bar_width*0.5, box_pos[2]])
    dummy_positions.append([-(box_length-bar_length)*0.5, box_width*0.5+bar_width*0.5, box_pos[2]])
    
    # Add dummy legs (attachment points)
    for i in range(len(dummy_positions)):
        x, y, z = dummy_positions[i]
        
        # Calculate position relative to body center
        rel_x = x - box_pos[0]
        rel_y = y - box_pos[1]
        rel_z = z - box_pos[2]
        
        # Add link
        urdf_content += """
    <link name="dummy_leg_{0}">
        <visual>
            <geometry>
                <box size="{1} {2} {3}"/>
            </geometry>
            <material name="light_blue"/>
        </visual>
        <collision>
            <geometry>
                <box size="{1} {2} {3}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="{4}"/>
            <inertia ixx="{5}" ixy="0.0" ixz="0.0" iyy="{6}" iyz="0.0" izz="{7}"/>
        </inertial>
    </link>
    
    <joint name="body_to_dummy_{0}" type="fixed">
        <parent link="base_link"/>
        <child link="dummy_leg_{0}"/>
        <origin xyz="{8} {9} {10}" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="0" upper="0" effort="10.0" velocity="0"/>
    </joint>
""".format(
            i, bar_length, bar_width, bar_height, 
            bar_mass,
            bar_mass * (bar_width**2 + bar_height**2) / 12.0,  # ixx
            bar_mass * (bar_length**2 + bar_height**2) / 12.0,  # iyy
            bar_mass * (bar_length**2 + bar_width**2) / 12.0,   # izz
            rel_x, rel_y, rel_z
        )
    
    # Define leg segments and joints
    leg_index = 0
    for leg_group in range(6):  # 6 legs (3 on each side)
        # Each leg has 3 segments
        for segment in range(3):
            if segment == 0:
                # First segment for each leg connects to the dummy leg
                parent_link = f"dummy_leg_{leg_group}"
                joint_type = "revolute"
                joint_axis = "0 -1 0"  # Rotate around y-axis (horizontal)
                dummy_index = leg_group
                
                # Position relative to the dummy leg
                x, y, z = leg_positions[leg_index]
                dx, dy, dz = dummy_positions[dummy_index]
                rel_x = x - dx
                rel_y = y - dy
                rel_z = z - dz
                
            else:
                # Other segments connect to the previous segment
                parent_link = f"leg_{leg_index-1}"
                joint_type = "revolute"
                joint_axis = "1 0 0"  # Rotate around x-axis (vertical)
                
                # Position relative to previous segment
                x, y, z = leg_positions[leg_index]
                prev_x, prev_y, prev_z = leg_positions[leg_index-1]
                rel_x = x - prev_x
                rel_y = y - prev_y
                rel_z = z - prev_z
            
            # Add link
            urdf_content += """
    <link name="leg_{0}">
        <visual>
            <geometry>
                <box size="{1} {2} {3}"/>
            </geometry>
            <material name="light_blue"/>
        </visual>
        <collision>
            <geometry>
                <box size="{1} {2} {3}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="{4}"/>
            <inertia ixx="{5}" ixy="0.0" ixz="0.0" iyy="{6}" iyz="0.0" izz="{7}"/>
        </inertial>
    </link>
    
    <joint name="joint_{0}" type="{8}">
        <parent link="{9}"/>
        <child link="leg_{0}"/>
        <origin xyz="{10} {11} {12}" rpy="0 0 0"/>
        <axis xyz="{13}"/>
        <limit lower="{14}" upper="{15}" effort="20.0" velocity="1.0"/>
    </joint>
""".format(
                leg_index, bar_length, bar_width, bar_height, 
                bar_mass,
                bar_mass * (bar_width**2 + bar_height**2) / 12.0,  # ixx
                bar_mass * (bar_length**2 + bar_height**2) / 12.0,  # iyy
                bar_mass * (bar_length**2 + bar_width**2) / 12.0,   # izz
                joint_type, parent_link,
                rel_x, rel_y, rel_z,
                joint_axis,
                -math.pi/2, math.pi/2  # Joint limits
            )
            
            leg_index += 1
    
    # Add a head to indicate the front of the robot
    head_length = 0.2  # Length (in x direction)
    head_width = 0.15  # Width (in y direction)
    head_height = 0.1  # Height (in z direction)
    head_mass = 0.1    # Lightweight
    
    # Position the head at the front of the body
    head_x = box_length / 2  # At the front edge of the body
    head_y = 0               # Centered horizontally
    head_z = box_height      # On top of the body
    
    urdf_content += """
    <link name="head">
        <visual>
            <geometry>
                <box size="{0} {1} {2}"/>
            </geometry>
            <material name="pink"/>
        </visual>
        <collision>
            <geometry>
                <box size="{0} {1} {2}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="{3}"/>
            <inertia ixx="{4}" ixy="0.0" ixz="0.0" iyy="{5}" iyz="0.0" izz="{6}"/>
        </inertial>
    </link>
    
    <joint name="body_to_head" type="fixed">
        <parent link="base_link"/>
        <child link="head"/>
        <origin xyz="{7} {8} {9}" rpy="0 0 0"/>
    </joint>
""".format(
        head_length, head_width, head_height,
        head_mass,
        head_mass * (head_width**2 + head_height**2) / 12.0,  # ixx
        head_mass * (head_length**2 + head_height**2) / 12.0,  # iyy
        head_mass * (head_length**2 + head_width**2) / 12.0,   # izz
        head_x, head_y, head_z
    )
    
    # Close the URDF file
    urdf_content += """
</robot>
"""
    
    return urdf_content

def save_urdf(urdf_content, filename="src/robot/urdf/legged_robot.urdf"):
    """Save the URDF content to a file."""
    with open(filename, 'w') as f:
        f.write(urdf_content)
    print(f"URDF saved to {filename}")

if __name__ == "__main__":
    urdf_content = generate_urdf()
    save_urdf(urdf_content)