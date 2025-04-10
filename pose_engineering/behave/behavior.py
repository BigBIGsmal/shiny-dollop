import os
import numpy as np
import math
import cv2

def classify_arm(angle):
    """
    Classify arm bend status based on angle
    Returns:
        status (str): Classification label
        color (tuple): BGR color for visualization
    """
    if angle > 160: 
        return "STRAIGHT", (0, 255, 0)  # Green
    elif angle > 90: 
        return "PARTIALLY BENT", (0, 255, 255)  # Yellow
    else: 
        return "FULLY BENT", (0, 0, 255)  # Red

def init_bend_counters():
    """Initialize bend tracking state"""
    return {
        'left_count': 0,
        'right_count': 0,
        'prev_left_status': None,
        'prev_right_status': None
    }
    
def update_bend_counters(counters, left_status, right_status):
    """
    Update bend counts when arm enters FULLY BENT state
    Returns updated counters
    """
    if left_status == "FULLY BENT" and counters['prev_left_status'] != "FULLY BENT":
        counters['left_count'] += 1
    if right_status == "FULLY BENT" and counters['prev_right_status'] != "FULLY BENT":
        counters['right_count'] += 1
    
    counters['prev_left_status'] = left_status
    counters['prev_right_status'] = right_status
    return counters

def calculate_angular_displacement(shoulder, elbow, wrist):
    """
    Calculate angular displacement of arm using elbow as pivot in 2D space
    Returns angle in degrees (0-180) and direction (up/down)
    """
    shoulder = np.array(shoulder)
    elbow = np.array(elbow)
    wrist = np.array(wrist)
    
    upper_arm = shoulder - elbow
    forearm = wrist - elbow
    
    cosine_angle = np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    cross = np.cross(upper_arm, forearm)
    direction = "up" if cross > 0 else "down"
    
    return angle, direction
    
def arm_analysis(arm_status, keypoint_data):
    if len(keypoint_data) > 0:
        kp_dict = {name: (x, y) for name, x, y in keypoint_data}
        required_kps = ['L_shoulder', 'L_elbow', 'L_wrist', 'R_shoulder', 'R_elbow', 'R_wrist']
        if all(kp in kp_dict for kp in required_kps):
            
            # Left arm analysis
            left_angle, left_dir = calculate_angular_displacement(
                kp_dict['L_shoulder'], kp_dict['L_elbow'], kp_dict['L_wrist'])
            left_status, left_color = classify_arm_bend_level(left_angle)
            arm_status['left'] = (left_status, left_color, left_angle, left_dir)

            # Right arm analysis
            right_angle, right_dir = calculate_angular_displacement(
                kp_dict['R_shoulder'], kp_dict['R_elbow'], kp_dict['R_wrist'])
            right_status, right_color = classify_arm_bend_level(right_angle)
            arm_status['right'] = (right_status, right_color, right_angle, right_dir)
            
            
def analyze_arms(keypoint_dict):
    """
    Perform complete arm analysis for both arms
    Returns:
        left_status, left_color, left_angle, left_dir
        right_status, right_color, right_angle, right_dir
    """
    # Left arm analysis
    left_angle, left_dir = calculate_angular_displacement(
        keypoint_dict['L_shoulder'],
        keypoint_dict['L_elbow'],
        keypoint_dict['L_wrist']
    )
    left_status, left_color = classify_arm(left_angle)
    
    # Right arm analysis
    right_angle, right_dir = calculate_angular_displacement(
        keypoint_dict['R_shoulder'],
        keypoint_dict['R_elbow'],
        keypoint_dict['R_wrist']
    )
    right_status, right_color = classify_arm(right_angle)
    
    return (
        left_status, left_color, left_angle, left_dir,
        right_status, right_color, right_angle, right_dir
    )

def draw_angle_arcs(frame, elbow_pos, angle, color, radius=30):
    """
    Draw angle visualization arcs at elbow positions
    """
    cv2.ellipse(
        frame,
        tuple(map(int, elbow_pos)),
        (radius, radius),
        0,  # Angle
        0,  # Start angle
        angle,  # End angle
        color,
        2  # Thickness
    )
    
def display_arm_status(frame, start_x, start_y, 
                      status, color, angle, direction, count):
    """
    Draw arm status information on frame
    Returns updated y-position for next display item
    """
    # Arm status
    cv2.putText(
        frame, 
        f"Arm: {status}", 
        (start_x, start_y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, color, 2
    )
    
    # Angle and direction
    cv2.putText(
        frame,
        f"Angle: {angle:.1f}° ({direction})",
        (start_x, start_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 1
    )
    
    # Bend count
    cv2.putText(
        frame,
        f"Bend Count: {count}",
        (start_x, start_y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2
    )
    
    return start_y + 90  # Return next y-position

# Initialize in your processing function
bend_counters = init_bend_counters()

# For each frame:
keypoint_dict = {name: (x, y) for name, x, y in keypoint_data}

# Analyze arms
(left_status, left_color, left_angle, left_dir,
 right_status, right_color, right_angle, right_dir) = analyze_arms(keypoint_dict)

# Update bend counters
bend_counters = update_bend_counters(
    bend_counters,
    left_status,
    right_status
)

# Draw angle arcs
draw_angle_arcs(frame, keypoint_dict['L_elbow'], left_angle, left_color)
draw_angle_arcs(frame, keypoint_dict['R_elbow'], right_angle, right_color)

# Display status
text_y = display_arm_status(
    frame, text_x, text_y,
    "Left " + left_status, left_color, 
    left_angle, left_dir, bend_counters['left_count']
)

text_y = display_arm_status(
    frame, text_x, text_y,
    "Right " + right_status, right_color,
    right_angle, right_dir, bend_counters['right_count']
)