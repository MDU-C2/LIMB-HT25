import numpy as np

def normalize_quaternion(q):
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm > 1e-8:
        return q / norm
    return np.array([1.0, 0.0, 0.0, 0.0])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    """Return quaternion conjugate (inverse rotation)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def madgwick_update(q, accel, gyro, dt, beta=0.1):
    """
    Madgwick filter update for IMU orientation estimation.
    
    Args:
        q: Initial quaternion [w, x, y, z]
        accel: Acceleration [ax, ay, az] in m/s²
        gyro: Angular velocity [wx, wy, wz] in rad/s
        dt: Time step in seconds
        beta: Filter gain
    
    Returns:
        Updated quaternion [w, x, y, z]
    """
    accel_norm = np.linalg.norm(accel)
    
    if accel_norm < 1e-8:
        # No acceleration, pure gyro integration
        omega_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
        q_dot = 0.5 * quaternion_multiply(q, omega_quat)
        q_new = q + q_dot * dt
        return normalize_quaternion(q_new)
    
    accel_normalized = accel / accel_norm
    
    # Compute objective function and Jacobian
    w, x, y, z = q
    
    f_g = np.array([
        2*(x*z - w*y) - accel_normalized[0],
        2*(w*x + y*z) - accel_normalized[1],
        2*(0.5 - x*x - y*y) - accel_normalized[2]
    ])
    
    J_g = np.array([
        [-2*y, 2*z, -2*w, 2*x],
        [2*x, 2*w, 2*z, 2*y],
        [0, -4*x, -4*y, 0]
    ])
    
    # Gradient descent step
    step = J_g.T @ f_g
    step_norm = np.linalg.norm(step)
    if step_norm > 1e-8:
        step = step / step_norm
    
    # Update quaternion
    omega_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
    q_dot_gyro = 0.5 * quaternion_multiply(q, omega_quat)
    q_dot_correction = -beta * step
    q_dot = q_dot_gyro + q_dot_correction
    q_new = q + q_dot * dt
    
    return normalize_quaternion(q_new)