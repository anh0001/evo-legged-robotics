import pybullet as p
import pybullet_data
import numpy as np
import time
import os


class Environment:
    """
    Enhanced PyBullet simulation environment with optimized physics parameters
    to prevent leg vibrations and improve stability.
    """
    
    def __init__(
        self,
        render=True,
        time_step=1.0/240.0,
        gravity=-9.81,
        terrain_type="flat",
        real_time=False,
        num_solver_iterations=100,
        num_sub_steps=2,
        enable_cone_friction=True,
        split_impulse_enabled=True,
        split_impulse_penetration_threshold=-0.02,
        contact_breaking_threshold=0.02,
        restitution_velocity_threshold=0.05,
        erp=0.2,
        contact_erp=0.1,
        friction_erp=0.8,
        global_cfm=1e-3
    ):
        """Initialize environment with enhanced physics parameters."""
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setRealTimeSimulation(1 if real_time else 0)
        self.real_time = real_time
        
        # Use Bullet recommended timestep for stability (1/240s)
        self.time_step = time_step
        p.setTimeStep(time_step)
        p.setGravity(0, 0, gravity)
        
        # Enhanced physics parameters to prevent vibrations
        p.setPhysicsEngineParameter(
            fixedTimeStep=time_step,
            numSolverIterations=num_solver_iterations,
            numSubSteps=num_sub_steps,
            erp=erp,
            contactERP=contact_erp,
            frictionERP=friction_erp,
            globalCFM=global_cfm,
            contactBreakingThreshold=contact_breaking_threshold,
            enableConeFriction=1 if enable_cone_friction else 0,
            useSplitImpulse=1 if split_impulse_enabled else 0,
            splitImpulsePenetrationThreshold=split_impulse_penetration_threshold,
            restitutionVelocityThreshold=restitution_velocity_threshold,
            deterministicOverlappingPairs=1,
            enableFileCaching=0
        )
        
        # Initialize simulation time and other attributes
        self.sim_time = 0.0
        self.start_time = time.time()
        self.robot_id = None
        self.objects = []
        self.terrain_type = terrain_type
        
        # Create ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = self._create_terrain(terrain_type)
        
        # Enhanced ground properties for stability
        p.changeDynamics(
            self.ground_id, -1,
            lateralFriction=2.0,
            spinningFriction=0.01,
            rollingFriction=0.001,
            restitution=0.0,
            contactDamping=30.0,
            contactStiffness=300.0,
            linearDamping=0.15,
            angularDamping=0.15
        )
        
        # Add obstacles if requested
        if terrain_type == "obstacles":
            self._add_obstacles()
        
        # Set up camera
        self._setup_camera()
        self.setup_debug_visualizer()
    
    def _create_terrain(self, terrain_type):
        """Create terrain based on specified type."""
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
            return ground_id
        
        else:
            return p.loadURDF("plane.urdf")
    
    def _generate_rough_terrain(self, width, height):
        """Generate a heightfield for rough terrain."""
        heightfield = np.zeros((width, height), dtype=np.float32)
        
        # Add some random noise with smoothing
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
        
        heightfield = heightfield.flatten()
        return heightfield
    
    def _add_obstacles(self, num_obstacles=25):
        """Add obstacles to the environment in a 5x5 grid pattern."""
        box_length = 0.4
        box_width = 0.4
        box_height = 0.1
        
        for i in range(5):
            for j in range(5):
                col_id = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[box_length/2, box_width/2, box_height/2]
                )
                
                vis_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[box_length/2, box_width/2, box_height/2],
                    rgbaColor=[1.0, 0.0, 1.0, 1.0]
                )
                
                x = (i - 2.0) + np.random.random() * 0.5
                y = (j - 2.0) + np.random.random() * 0.5
                z = box_height / 2
                
                obstacle_id = p.createMultiBody(
                    baseMass=1.0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[x, y, z]
                )
                
                # Set obstacle properties to prevent bouncing
                p.changeDynamics(
                    obstacle_id, -1,
                    lateralFriction=0.8,
                    restitution=0.1,
                    contactDamping=50.0,
                    contactStiffness=3000.0
                )
                
                self.objects.append(obstacle_id)
    
    def _setup_camera(self):
        """Set up the camera for visualization."""
        self.cam_distance = 3.0
        self.cam_yaw = 101.0
        self.cam_pitch = -27.5
        self.cam_target_pos = [0, 0, 0.5]
        
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
    
    def add_robot(self, robot):
        """Add a robot to the environment."""
        self.robot_id = robot.body_id
        
        # Apply enhanced dynamics to robot links
        for joint_idx in range(p.getNumJoints(robot.body_id)):
            p.changeDynamics(
                robot.body_id, joint_idx,
                lateralFriction=2.0,
                spinningFriction=0.01,
                rollingFriction=0.001,
                restitution=0.0,
                contactDamping=200.0,
                contactStiffness=2500.0,
                jointDamping=0.05,
                linearDamping=0.1,
                angularDamping=0.1
            )
    
    def step(self, actions=None):
        """Step the simulation forward."""
        if actions is not None:
            self.apply_actions(actions)
        
        if not self.real_time:
            p.stepSimulation()
            self.sim_time += self.time_step
        else:
            self.sim_time = time.time() - self.start_time
        
        if self.robot_id is not None:
            self.update_camera()
        
        return self.get_observation()
    
    def apply_actions(self, actions):
        """Apply actions to the robot."""
        pass
    
    def update_camera(self):
        """Update camera position to follow the robot."""
        if self.robot_id is None:
            return
        
        try:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            self.cam_target_pos = [pos[0], pos[1], 0.5]
            
            p.resetDebugVisualizerCamera(
                cameraDistance=self.cam_distance,
                cameraYaw=self.cam_yaw,
                cameraPitch=self.cam_pitch,
                cameraTargetPosition=self.cam_target_pos
            )
        except:
            pass
    
    def get_observation(self):
        """Get current observation from the environment."""
        obs = {'time': self.sim_time}
        
        if self.robot_id is not None:
            try:
                pos, orn = p.getBasePositionAndOrientation(self.robot_id)
                rot_matrix = p.getMatrixFromQuaternion(orn)
                
                obs.update({
                    'robot_position': pos,
                    'robot_orientation': orn,
                    'robot_rotation_matrix': rot_matrix
                })
            except:
                pass
        
        return obs
    
    def reset(self):
        """Reset the simulation environment."""
        self.sim_time = 0
        self.start_time = time.time()
        
        for obj_id in self.objects:
            try:
                p.removeBody(obj_id)
            except:
                pass
        self.objects = []
        
        try:
            p.removeBody(self.ground_id)
        except:
            pass
        self.ground_id = self._create_terrain(self.terrain_type)
        
        self._setup_camera()
        return self.get_observation()
    
    def close(self):
        """Close the simulation."""
        p.disconnect(self.client)
