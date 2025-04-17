import numpy as np
import math


def radian_to_degree(angle):
    """
    Convert angle from radians to degrees.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Angle in degrees
    """
    return angle * 180.0 / math.pi


def degree_to_radian(angle):
    """
    Convert angle from degrees to radians.
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Angle in radians
    """
    return angle * math.pi / 180.0


def random_uniform():
    """
    Generate a random number from uniform distribution [0, 1).
    
    Returns:
        Random number between 0 (inclusive) and 1 (exclusive)
    """
    return np.random.random()


def random_normal():
    """
    Generate a random number from normal distribution using central limit theorem.
    Approximates the normal random number generation from the original C++ code.
    
    Returns:
        Random number from approximated normal distribution
    """
    return (np.random.random() + np.random.random() + np.random.random() +
            np.random.random() + np.random.random() + np.random.random() +
            np.random.random() + np.random.random() + np.random.random() +
            np.random.random() + np.random.random() + np.random.random() - 6.0)


def rotation_matrix_to_euler(rotation_matrix):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    
    Args:
        rotation_matrix: 3x3 rotation matrix or 9-element list/array
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    if isinstance(rotation_matrix, list):
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
    
    # Handle singularity (gimbal lock)
    if abs(rotation_matrix[2, 0]) >= 1.0:
        # Gimbal lock case
        yaw = 0
        if rotation_matrix[2, 0] < 0:
            pitch = math.pi / 2
            roll = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
        else:
            pitch = -math.pi / 2
            roll = math.atan2(-rotation_matrix[0, 1], -rotation_matrix[0, 2])
    else:
        # Regular case
        pitch = -math.asin(rotation_matrix[2, 0])
        roll = math.atan2(rotation_matrix[2, 1] / math.cos(pitch),
                          rotation_matrix[2, 2] / math.cos(pitch))
        yaw = math.atan2(rotation_matrix[1, 0] / math.cos(pitch),
                         rotation_matrix[0, 0] / math.cos(pitch))
    
    return (roll, pitch, yaw)


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        roll: Rotation around x-axis in radians
        pitch: Rotation around y-axis in radians
        yaw: Rotation around z-axis in radians
        
    Returns:
        3x3 rotation matrix
    """
    # Calculate rotation components
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
        [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
        [-sin_p, cos_p*sin_r, cos_p*cos_r]
    ])
    
    return rotation_matrix


def quaternion_to_euler(quaternion):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    x, y, z, w = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (roll, pitch, yaw)


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll: Rotation around x-axis in radians
        pitch: Rotation around y-axis in radians
        yaw: Rotation around z-axis in radians
        
    Returns:
        Quaternion [x, y, z, w]
    """
    # Calculate half angles
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    # Calculate quaternion components
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    
    return [x, y, z, w]


def homogeneous_transform(translation, rotation):
    """
    Create a 4x4 homogeneous transformation matrix.
    
    Args:
        translation: 3D translation vector [x, y, z]
        rotation: 3x3 rotation matrix or quaternion [x, y, z, w]
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    transform = np.eye(4)
    
    # Set translation
    transform[:3, 3] = translation
    
    # Set rotation
    if len(rotation) == 4:
        # If quaternion, convert to rotation matrix
        x, y, z, w = rotation
        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        transform[:3, :3] = rotation_matrix
    else:
        # If rotation matrix, directly use it
        transform[:3, :3] = rotation
    
    return transform


def transform_point(transform, point):
    """
    Transform a point using a homogeneous transformation matrix.
    
    Args:
        transform: 4x4 homogeneous transformation matrix
        point: 3D point [x, y, z]
        
    Returns:
        Transformed 3D point
    """
    # Convert to homogeneous coordinates
    homogeneous_point = np.append(point, 1)
    
    # Apply transformation
    transformed_point = np.dot(transform, homogeneous_point)
    
    # Convert back to 3D coordinates
    return transformed_point[:3]


def angle_difference(angle1, angle2):
    """
    Calculate the smallest difference between two angles in radians.
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
        
    Returns:
        Angle difference in range [-pi, pi]
    """
    diff = angle1 - angle2
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def gaussian_membership(x, center, width=1.0):
    """
    Calculate Gaussian membership function value.
    
    Args:
        x: Input value
        center: Center of the Gaussian
        width: Width parameter (controls the spread)
        
    Returns:
        Gaussian membership value
    """
    return math.exp(-((x - center) ** 2) / (width ** 2))


def sigmoid(x):
    """
    Calculate sigmoid function value.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid function value
    """
    return 1.0 / (1.0 + math.exp(-x))