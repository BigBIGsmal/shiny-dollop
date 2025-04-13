# features.py
import numpy as np
from collections import deque
import cv2

def init_feature_state():
    """Initialize state management for temporal features"""
    return {
        'prev_positions': {},
        'displacement_history': {
            'L_wrist': deque(maxlen=5),
            'R_wrist': deque(maxlen=5),
            'L_elbow': deque(maxlen=5),
            'R_elbow': deque(maxlen=5)
        },
        'prev_velocities': {}
    }

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def calculate_pose_stability(keypoints):
    """Calculate comprehensive pose stability index"""
    distances = [
        calculate_distance(keypoints['Head'], keypoints['L_shoulder']),
        calculate_distance(keypoints['Head'], keypoints['R_shoulder']),
        calculate_distance(keypoints['Head'], keypoints['L_wrist']),
        calculate_distance(keypoints['Head'], keypoints['R_wrist']),
        calculate_distance(keypoints['L_elbow'], keypoints['L_wrist']),
        calculate_distance(keypoints['R_elbow'], keypoints['R_wrist']),
        calculate_distance(keypoints['L_shoulder'], keypoints['R_shoulder'])
    ]
    return np.mean(distances)

def update_temporal_features(state, current_positions, delta_time):
    """Calculate and update temporal features (velocity, acceleration)"""
    features = {}
    
    for joint in ['L_wrist', 'R_wrist', 'L_elbow', 'R_elbow']:
        # Displacement
        disp = calculate_distance(current_positions[joint], 
                                 state['prev_positions'].get(joint, (0, 0)))
        state['displacement_history'][joint].append(disp)
        
        # Velocity
        vel = disp / delta_time if delta_time != 0 else 0
        features[f'{joint}_velocity'] = vel
        
        # Acceleration
        prev_vel = state['prev_velocities'].get(joint, 0)
        accel = (vel - prev_vel) / delta_time if delta_time != 0 else 0
        features[f'{joint}_acceleration'] = accel
        
        # Update state
        state['prev_velocities'][joint] = vel
    
    state['prev_positions'] = current_positions
    return features

def draw_features(frame, features):
    """Draw calculated features on frame"""
    y_offset = 400  # Start lower to avoid overlap
    x_offset = 20 
    
    # Draw header
    cv2.putText(frame, "Engineered Features:", (x_offset, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    y_offset += 40

    # Draw spatial features
    for feat in ['head_to_l_wrist', 'head_to_r_wrist', 'wrist_distance', 'pose_stability']:
        if feat in features:
            cv2.putText(frame, f"{feat}: {features[feat]:.2f}", 
                       (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0,255,0), 2)
            y_offset += 30

    # Draw temporal features
    for joint in ['L_wrist', 'R_wrist', 'L_elbow', 'R_elbow']:
        cv2.putText(frame, f"{joint} Velocity: {features.get(f'{joint}_velocity', 0):.2f}", 
                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255,255,0), 2)
        y_offset += 30
        cv2.putText(frame, f"{joint} Acceleration: {features.get(f'{joint}_acceleration', 0):.2f}", 
                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255,255,0), 2)
        y_offset += 30

    return frame