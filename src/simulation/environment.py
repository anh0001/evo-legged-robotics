import pybullet as p
import pybullet_data
import numpy as np
import time
import os


class Environment:
    """
    PyBullet simulation environment for the legged robot.
    This is a port of the ODE-based environment from the original C++ code.
    """
    
    def __init__(self, render=True, time_step=0.01, gravity=-9.81,
                 erp=0.2, cfm=0.00001, terrain_type="flat"):
        """
        Initialize the simulation environment.
        
        Args:
            render: Whether to render the simulation
            time_step: Simulation time step
            gravity: Gravity acceleration value
            erp: Error Reduction Parameter (similar to ODE's ERP)
            cfm: Constraint Force Mixing (similar to ODE's CFM)
            terrain_type: Type of terrain to generate ("flat", "rough", "obstacles")
        """
        # Connect to physics server
        self.client = p.connect(p.GUI if render else p.DIRECT)
        
        # Configure simulation parameters
        self.time_step = time_step
        p.setTimeStep(time_step)
        p.setGravity(0, 0, gravity)
        
        # Set additional physics parameters
        p.setPhysicsEngineParameter(
            fixedTimeStep=time_step,
            numSolverIterations=50,
            numSubSteps=4,
            erp=erp,
            contactERP=erp,
            frictionERP=erp,
            globalCFM=cfm
        )
        
        # Add data path for URDF models
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Create ground
        self.terrain_type = terrain_type
        self.ground_id = self._create_terrain(terrain_type)
        
        # Store robot IDs
        self.robot_id = None
        self.objects = []
        
        # Set up camera
        self._setup_camera()
        
        # For timing
        self.start_time = time.time()
        self.sim_time = 0
        
        # For visualization
        if render:
            self.setup_debug_visualizer()
    
    def _create_terrain(self, terrain_type):
        """
        Create terrain based on specified type.
        
        Args:
            terrain_type: Type of terrain to generate
            
        Returns:
            ID of the created terrain
        """
        if terrain_type == "flat":
            return p.loadURDF("plane.urdf")
        
        elif terrain_type == "rough":
            # Create a heightfield terrain
            terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[0.05, 0.05, 1],
                heightfieldTextureScaling=128,
                heightfieldData=self._generate_rough_terrain(256, 256),
                numHeightfieldRows=256,
                numHeightfieldColumns=256
            )
            terrain = p.createMultiBody(0, terrain_shape)
            p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])
            return terrain
        
        elif terrain_type == "obstacles":
            # Create a flat ground with obstacles
            ground_id = p.loadURDF("plane.urdf")
            
            # Add various obstacles
            self._add_obstacles()
            
            return ground_id
        
        else:
            # Default to flat terrain
            return p.loadURDF("plane.urdf")
    
    def _generate_rough_terrain(self, width, height):
        """
        Generate a heightfield for rough terrain.
        
        Args:
            width: Width of the heightfield
            height: Height of the heightfield
            
        Returns:
            Heightfield data
        """
        # Generate a random heightfield with some smoothing
        heightfield = np.zeros((width, height), dtype=np.float32)
        
        # Add some random noise
        for i in range(width):
            for j in range(height):
                # Base height
                heightfield[i, j] = 0
                
                # Add perlin-like noise (simplified)
                freq = 0.1
                heightfield[i, j] += 0.1 * np.sin(i * freq) * np.cos(j * freq)
                heightfield[i, j] += 0.05 * np.sin(i * freq * 2) * np.cos(j * freq * 2)
                heightfield[i, j] += 0.025 * np.sin(i * freq * 4) * np.cos(j * freq * 4)
                
                # Add some random bumps
                if np.random.random() < 0.01:
                    for di in range(-3, 4):
                        for dj in range(-3, 4):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < width and 0 <= nj < height:
                                dist = np.sqrt(di**2 + dj**2)
                                if dist < 3:
                                    heightfield[ni, nj] += 0.1 * (1 - dist/3)
        
        # Scale the heightfield
        heightfield = heightfield.flatten()
        return heightfield
    
    def _add_obstacles(self, num_obstacles=25):
        """
        Add obstacles to the environment.
        
        Args:
            num_obstacles: Number of obstacles to add
        """
        # Box dimensions
        box_length = 0.4
        box_width = 0.4
        box_height = 0.1
        
        # Create obstacles in a grid pattern with some randomization
        for i in range(5):
            for j in range(5):
                # Create box collision shape
                col_id = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[box_length/2, box_width/2, box_height/2]
                )
                
                # Create visual shape
                vis_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[box_length/2, box_width/2, box_height/2],
                    rgbaColor=[1.0, 0.0, 1.0, 1.0]
                )
                
                # Create multibody
                x = (i - 2.0) + np.random.random() * 0.5
                y = (j - 2.0) + np.random.random() * 0.5
                z = box_height / 2
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,  # Static obstacle
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[x, y, z]
                )
                
                self.objects.append(obstacle_id)
    
    def _setup_camera(self):
        """Set up the camera for visualization."""
        # Set camera parameters
        self.cam_distance = 3.0
        self.cam_yaw = 101.0
        self.cam_pitch = -27.5
        self.cam_target_pos = [0, 0, 0.5]
        
        # Apply camera settings
        p.resetDebugVisualizerCamera(
            cameraDistance=self.cam_distance,
            cameraYaw=self.cam_yaw,
            cameraPitch=self.cam_pitch,
            cameraTargetPosition=self.cam_target_pos
        )
    
    def setup_debug_visualizer(self):
        """Configure the debug visualizer settings."""
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
    def reset(self):
        """Reset the simulation environment."""
        # Reset simulation time
        self.sim_time = 0
        self.start_time = time.time()
        
        # Clear existing objects except ground
        for obj_id in self.objects:
            p.removeBody(obj_id)
        self.objects = []
        
        # Recreate terrain if needed
        p.removeBody(self.ground_id)
        self.ground_id = self._create_terrain(self.terrain_type)
        
        # Reset camera
        self._setup_camera()
        
        # Return initial observation
        return self.get_observation()
    
    def step(self, actions=None):
        """
        Step the simulation forward.
        
        Args:
            actions: Actions to apply (if any)
            
        Returns:
            Observation after step
        """
        # Apply actions if provided
        if actions is not None:
            self.apply_actions(actions)
        
        # Step the simulation
        p.stepSimulation()
        
        # Update simulation time
        self.sim_time += self.time_step
        
        # Update camera if robot has moved
        if self.robot_id is not None:
            self.update_camera()
        
        # Return observation
        return self.get_observation()
    
    def apply_actions(self, actions):
        """
        Apply actions to the robot.
        
        Args:
            actions: Actions to apply
        """
        # This function would typically set motor commands
        # The implementation depends on how actions are represented
        # For now, we assume no actions need to be applied
        pass
    
    def update_camera(self):
        """Update camera position to follow the robot."""
        if self.robot_id is None:
            return
        
        # Get robot position
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # Update camera target position
        self.cam_target_pos = [pos[0], pos[1], 0.5]
        
        # Apply camera settings
        p.resetDebugVisualizerCamera(
            cameraDistance=self.cam_distance,
            cameraYaw=self.cam_yaw,
            cameraPitch=self.cam_pitch,
            cameraTargetPosition=self.cam_target_pos
        )
    
    def get_observation(self):
        """
        Get current observation from the environment.
        
        Returns:
            Dictionary containing environment state
        """
        # Basic observation with time
        obs = {
            'time': self.sim_time,
        }
        
        # Add robot state if robot exists
        if self.robot_id is not None:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            rot_matrix = p.getMatrixFromQuaternion(orn)
            
            obs.update({
                'robot_position': pos,
                'robot_orientation': orn,
                'robot_rotation_matrix': rot_matrix
            })
        
        return obs
    
    def add_robot(self, robot):
        """
        Add a robot to the environment.
        
        Args:
            robot: Robot instance to add
        """
        self.robot_id = robot.body_id
    
    def close(self):
        """Close the simulation."""
        p.disconnect(self.client)


class TrainingEnvironment(Environment):
    """
    Extended environment for training with fitness evaluation and parallel simulations.
    """
    
    def __init__(self, render=False, time_step=0.01, gravity=-9.81,
                 terrain_type="flat"):
        """
        Initialize the training environment.
        
        Args:
            render: Whether to render the simulation
            time_step: Simulation time step
            gravity: Gravity acceleration value
            terrain_type: Type of terrain to generate
        """
        super().__init__(render, time_step, gravity, terrain_type=terrain_type)
        
        # Parameters for fitness evaluation
        self.max_steps = 1000
        self.current_step = 0
        
        # For storing trajectory data
        self.trajectory = []
    
    def reset(self):
        """Reset the environment for a new evaluation."""
        obs = super().reset()
        self.current_step = 0
        self.trajectory = []
        return obs
    
    def step(self, actions=None):
        """
        Step the training environment.
        
        Args:
            actions: Actions to apply
            
        Returns:
            (observation, reward, done, info)
        """
        # Step the simulation
        obs = super().step(actions)
        
        # Increment step counter
        self.current_step += 1
        
        # Record trajectory
        if self.robot_id is not None:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            self.trajectory.append((pos, orn))
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Calculate reward (simple version - can be extended)
        reward = self.calculate_reward()
        
        # Additional info
        info = {
            'steps': self.current_step,
            'trajectory_length': len(self.trajectory)
        }
        
        return obs, reward, done, info
    
    def calculate_reward(self):
        """
        Calculate reward based on robot state.
        
        Returns:
            Reward value
        """
        # Simple reward - distance traveled in x direction
        if len(self.trajectory) >= 2:
            start_pos = self.trajectory[0][0]
            current_pos = self.trajectory[-1][0]
            
            # Distance traveled in x-y plane
            distance = np.sqrt((current_pos[0] - start_pos[0])**2 + 
                               (current_pos[1] - start_pos[1])**2)
            
            return distance
        
        return 0.0
    
    def get_fitness(self):
        """
        Calculate fitness values for evolution.
        
        Returns:
            Dictionary of fitness metrics
        """
        if len(self.trajectory) < 2:
            return {
                'forward_distance': 0.0,
                'rotation': 0.0,
                'direction_alignment': 0.0,
                'energy_efficiency': 0.0
            }
        
        # Get initial and final states
        initial_pos, initial_orn = self.trajectory[0]
        final_pos, final_orn = self.trajectory[-1]
        
        # Calculate displacement
        displacement = np.array(final_pos) - np.array(initial_pos)
        distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
        
        # Calculate rotation change
        initial_rot_matrix = np.array(p.getMatrixFromQuaternion(initial_orn)).reshape(3, 3)
        final_rot_matrix = np.array(p.getMatrixFromQuaternion(final_orn)).reshape(3, 3)
        
        initial_direction = initial_rot_matrix[:, 0]  # First column is x-axis
        final_direction = final_rot_matrix[:, 0]
        
        # Calculate angle between initial and final direction
        dot_product = np.dot(initial_direction[:2], final_direction[:2])
        angle_change = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Calculate direction alignment
        direction_alignment = 0
        if distance > 0:
            forward_vector = displacement / distance
            alignment = np.dot(final_direction[:2], forward_vector[:2])
            direction_alignment = alignment
        
        # Calculate energy (placeholder - would need joint torques and velocities)
        # For now, we use a simple proxy based on distance vs. number of steps
        energy_efficiency = distance / self.current_step if self.current_step > 0 else 0
        
        return {
            'forward_distance': distance,
            'rotation': angle_change,
            'direction_alignment': direction_alignment,
            'energy_efficiency': energy_efficiency
        }
    
    def render_trajectory(self, filename=None):
        """
        Render the trajectory of the robot.
        
        Args:
            filename: If provided, save the rendering to this file
        """
        if len(self.trajectory) < 2:
            return
        
        # Create a list of points for the trajectory
        points = [pos for pos, _ in self.trajectory]
        
        # Render the trajectory
        line_color = [1, 0, 0]
        for i in range(len(points) - 1):
            p.addUserDebugLine(points[i], points[i+1], line_color, 2, lifeTime=0)
        
        # Save screenshot if filename is provided
        if filename is not None:
            p.getCameraImage(
                width=1024,
                height=768,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                shadow=1,
                lightDirection=[1, 1, 1],
                projectiveTextureView=0
            )
            # Note: PyBullet doesn't directly support saving screenshots
            # You would need to save the returned image