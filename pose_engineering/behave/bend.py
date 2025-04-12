# behavior.py (updated)

import numpy as np
import cv2

def classify_arm_bend_level(angle):
    """Classify arm bend status based on angle"""
    if angle > 160: 
        return "STRAIGHT", (0, 255, 0)  # Green
    elif angle > 90: 
        return "PARTIALLY BENT", (0, 255, 255)  # Yellow
    else: 
        return "FULLY BENT", (0, 0, 255)  # Red

def calculate_angular_displacement(shoulder, elbow, wrist):
    """Calculate angle between shoulder-elbow-wrist"""
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

def analyze_frame(keypoint_data, counters):
    """
    Main function to analyze arm behavior for a frame
    Returns:
        - analysis_results: Dictionary with arm data
        - updated counters
        - visualization elements
    """
    analysis_results = {
        'left': {},
        'right': {}
    }
    
    if not keypoint_data:
        return analysis_results, counters, None
    
    kp_dict = {name: (x, y) for name, x, y in keypoint_data}
    required_kps = ['L_shoulder', 'L_elbow', 'L_wrist', 'R_shoulder', 'R_elbow', 'R_wrist']
    
    if not all(kp in kp_dict for kp in required_kps):
        return analysis_results, counters, None
    
    # Left arm analysis
    left_angle, left_dir = calculate_angular_displacement(
        kp_dict['L_shoulder'], kp_dict['L_elbow'], kp_dict['L_wrist'])
    left_status, left_color = classify_arm_bend_level(left_angle)
    
    # Right arm analysis
    right_angle, right_dir = calculate_angular_displacement(
        kp_dict['R_shoulder'], kp_dict['R_elbow'], kp_dict['R_wrist'])
    right_status, right_color = classify_arm_bend_level(right_angle)
    
    left_status, left_color = classify_arm_bend_level(left_angle)
    right_status, right_color = classify_arm_bend_level(right_angle)
    # Update counters
    if left_status == "FULLY BENT" and counters['prev_left_status'] != "FULLY BENT":
        counters['left_count'] += 1
    if right_status == "FULLY BENT" and counters['prev_right_status'] != "FULLY BENT":
        counters['right_count'] += 1
    
    counters['prev_left_status'] = left_status
    counters['prev_right_status'] = right_status
    
    # Prepare visualization elements
    vis_elements = {
        'left_elbow': kp_dict['L_elbow'],
        'right_elbow': kp_dict['R_elbow'],
        'left_color': left_color,
        'right_color': right_color,
        'left_count': counters['left_count'],
        'right_count': counters['right_count']
    }
    
    # Prepare analysis results
    analysis_results['left'].update({
        'status': left_status,
        'angle': left_angle,
        'direction': left_dir,
        'count': counters['left_count']
    })
    
    analysis_results['right'].update({
        'status': right_status,
        'angle': right_angle,
        'direction': right_dir,
        'count': counters['right_count']
    })
    
    return analysis_results, counters, vis_elements

def draw_behavior_visuals(frame, vis_elements, analysis_results, counters):
    """Draw behavior analysis visuals on the frame"""
    if not vis_elements:
        return frame
    
    annotated_frame = frame.copy()

    # Access the entire "left" sub-dictionary
    left_data = analysis_results['left']

    # Access individual values under "left"
    left_status = analysis_results['left']['status']       # Value: 'STRAIGHT'
    left_angle = analysis_results['left']['angle']         # Value: np.float32(172.07626)
    left_direction = analysis_results['left']['direction'] # Value: 'up'
    left_count = analysis_results['left']['count']   
    
    # Access the entire "right" sub-dictionary
    right_data = analysis_results['right']

    # Access individual values under "right"
    right_status = analysis_results['right']['status']       # Value: 'PARTIALLY BENT'
    right_angle = analysis_results['right']['angle']         # Value: np.float32(136.06111)
    right_direction = analysis_results['right']['direction'] # Value: 'down'
    right_count = analysis_results['right']['count']   
    # Draw angle arcs
    cv2.ellipse(
        annotated_frame,
        tuple(map(int, vis_elements['left_elbow'])),
        (30, 30), 0, 0, analysis_results['left']['angle'],
        vis_elements['left_color'], 2
    )
    cv2.ellipse(
        annotated_frame,
        tuple(map(int, vis_elements['right_elbow'])),
        (30, 30), 0, 0,  analysis_results['right']['angle'],
        vis_elements['right_color'], 2
    )
    
    # Draw arm status info
    text_x = 20
    text_y = 30
    # Left arm info
    cv2.putText(
        annotated_frame,
        f"Left: {analysis_results['left']['status']}",
        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        vis_elements['left_color'], 2
    )
    cv2.putText(
        annotated_frame,
        f"Angle: {analysis_results['left']['angle']:.1f}°",
        (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        vis_elements['left_color'], 1
    )
    cv2.putText(
        annotated_frame,
        f"Count: {vis_elements['left_count']}",
        (text_x, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0), 2
    )
    
    # Right arm info
    cv2.putText(
        annotated_frame,
        f"Right: {analysis_results['right']['status']}",
        (text_x, text_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        vis_elements['right_color'], 2
    )
    cv2.putText(
        annotated_frame,
        f"Angle: {analysis_results['right']['angle']:.1f}°",
        (text_x, text_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        vis_elements['right_color'], 1
    )
    cv2.putText(
        annotated_frame,
        f"Count: {vis_elements['right_count']}",
        (text_x, text_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0), 2
    )
    
    return annotated_frame