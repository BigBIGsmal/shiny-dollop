import cv2
import numpy as np
from collections import defaultdict, deque

# State management
state = {
    'path_history': defaultdict(list),
    'total_distance': defaultdict(float),
    'prev_positions': {},
    'velocity_history': defaultdict(lambda: deque(maxlen=HALF_SECOND_WINDOW)),
    'frame_count': 0,
    'csv_data': []
}
HALF_SECOND_WINDOW = 15  # Frames in 0.5 seconds at 30fps

def analyze_dist(annotated_frame, keypoint_data):
    """
    Analyze distance between keypoints and update the state.
    Returns:
        - analysis_results: Dictionary with distance analysis results
        - acceleration
    """
    TRACKED_JOINTS = ['L_wrist', 'R_wrist']  # Joints to track
        # Convert keypoint_data list to a dictionary: {joint_name: (x, y)}
    # In analyze_dist():
    kp_dict = {name: (float(x), float(y)) for name, x, y in keypoint_data} if keypoint_data else {}
    print(f"Keypoint_data: {keypoint_data} \n")
    print(f"Keypoint_dict: {kp_dict} \n")
    # Track each joint's coordinates individually
    current_positions = {joint: kp_dict.get(joint) for joint in TRACKED_JOINTS}
    print(f"Current positions: {current_positions} \n")
    
    update_motion_tracking(current_positions, TRACKED_JOINTS)
    
 
    acceleration_ave = calculate_average_velocity(TRACKED_JOINTS)
    
    annotated_frame = draw_motion_visualization(annotated_frame, current_positions, TRACKED_JOINTS, acceleration_ave)
    

    return current_positions, acceleration_ave, annotated_frame

def draw_motion_visualization(frame, current_positions, TRACKED_JOINTS, acceleration_ave):
    """Visualize joint paths, distances, and velocities"""
    if not current_positions:
        return frame
    # Draw motion paths
    for joint in TRACKED_JOINTS:
        if len(state['path_history'][joint]) > 1:
            color = (0, 255, 255) if joint == 'L_wrist' else (255, 0, 255)  # Yellow/cyan paths
            for i in range(1, len(state['path_history'][joint])):
                cv2.line(frame, 
                        tuple(map(int, state['path_history'][joint][i-1])),
                        tuple(map(int, state['path_history'][joint][i])),
                        color, 2)
    
    # Draw current positions and coordinates
    print("Current positions:", current_positions)
    for joint, pos in current_positions.items():
        if pos:
            cv2.circle(frame, tuple(map(int, pos)), 8, (0, 255, 0), -1)
            coord_text = f"{joint}: ({int(pos[0])}, {int(pos[1])})"
            cv2.putText(frame, coord_text, (int(pos[0])+10, int(pos[1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Calculate average velocities
    avg_velocities = acceleration_ave
    
    # Display metrics
    # Display metrics - Start lower and add more spacing
    y_offset = 210  # Start below bend analysis text
    x_offset = 20  # Left margin
    for joint in TRACKED_JOINTS:
        # Distance counter
        cv2.putText(frame, f"{joint} Distance: {state['total_distance'][joint]:.1f} px", 
                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        y_offset += 35  # Increased spacing
        
        # Current coordinates
        pos = current_positions.get(joint)
        if pos:
            coord_text = f"{joint} Position: ({int(pos[0])}, {int(pos[1])})"
            cv2.putText(frame, coord_text, (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_offset += 30  # Increased spacing
        
        # Average velocity
        cv2.putText(frame, f"{joint} Speed: {avg_velocities[joint]:.1f} px/0.5s", 
                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255,255,0), 1)
        y_offset += 30  # Increased spacing
    
    return frame
def calculate_average_velocity(TRACKED_JOINTS):
    """Calculate average velocity over last 0.5 seconds for each joint"""
    avg_velocities = {}
    for joint in TRACKED_JOINTS:
        velocities = list(state['velocity_history'][joint])
        if velocities:
            avg_velocities[joint] = sum(velocities) / len(velocities)
        else:
            avg_velocities[joint] = 0
    return avg_velocities
def update_motion_tracking(current_keypoint_positions, TRACKED_JOINTS):
    """Track joint paths, distances, and velocities"""
    
    for joint in TRACKED_JOINTS:
        current_pos = current_keypoint_positions.get(joint)
        print(f"Current_pos for {joint}: {current_pos} \n")
        
        if current_pos:
            print("inside if current_pos")
            # Store position for path drawing
            state['path_history'][joint].append(current_pos)
            print(f"Path history for {joint}: {state['path_history'][joint]} \n")
            # Update total distance traveled
            print(f"State['prev_positions']: {state['prev_positions']} \n")
            if joint in state['prev_positions']:
                print("inside if joint in state['prev_positions']")
                step_distance = calculate_distance(state['prev_positions'][joint], current_pos)
                state['total_distance'][joint] += step_distance
                state['velocity_history'][joint].append(step_distance)
                
            state['prev_positions'][joint] = current_pos
            print("State Saved?")
            print(f"current_pos: {current_pos} \n")
            
           
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    try:
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    except (TypeError, IndexError):
        return 0