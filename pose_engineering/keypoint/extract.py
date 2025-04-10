import os
import cv2
import pandas as pd
import numpy as np
from behave.behavior import arm_analysis, calculate_angular_displacement, classify_arm_bend_level


def get_keypoint_coordinates(results, frame_width, frame_height, keypoints):
    """
    Extract keypoint coordinates from YOLOv8 results
    Returns:
        - List of (keypoint_name, x, y) tuples
        - List of bounding box coordinates [x1, y1, x2, y2]
    """
    keypoint_data = []
    bounding_box = []
    
    if results and results[0].keypoints is not None:
        keypoints_data = results[0].keypoints.xy.cpu().numpy()
        
        if keypoints_data.shape[0] > 0:
            for i, (x, y) in enumerate(keypoints_data[0]):
                if i < len(keypoints):
                    # Normalize coordinates if needed (0-1 to pixel values)
                    x_norm = x * frame_width if x <= 1 else x
                    y_norm = y * frame_height if y <= 1 else y
                    keypoint_data.append((keypoints[i], x_norm, y_norm))
        
        if results[0].boxes is not None:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            if len(bboxes) > 0:
                bounding_box = list(map(int, bboxes[0]))  # Use first detected bbox
    
    return keypoint_data, bounding_box

def draw_skeleton(frame, keypoint_data, bbox=None, connections=[]):
    """
    Draw keypoints, skeleton, and bounding box on frame
    Returns annotated frame
    """
    annotated_frame = frame.copy()
    
    # Draw bounding box
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw keypoints and skeleton
    if keypoint_data:
        # Draw keypoints
        for key, x, y in keypoint_data:
            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(annotated_frame, key, (int(x) + 5, int(y) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw skeleton connections
        kp_dict = {k: (x, y) for k, x, y in keypoint_data}
        for connection in connections:
            if connection[0] in kp_dict and connection[1] in kp_dict:
                pt1 = tuple(map(int, kp_dict[connection[0]]))
                pt2 = tuple(map(int, kp_dict[connection[1]]))
                cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 2)
    
    return annotated_frame

def display_keypoint_info(frame, keypoint_data, frame_num):
    """
    Add keypoint coordinates and frame info to the frame
    Returns annotated frame
    """
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    text_y = 30
    text_x = width - 300  # Right side
    
    # Frame number
    cv2.putText(annotated_frame, f"Frame: {frame_num}", (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    text_y += 30
    
    # Keypoint coordinates
    for key, x, y in keypoint_data:
        coord_text = f"{key}: ({int(x)}, {int(y)})"
        cv2.putText(annotated_frame, coord_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        text_y += 20
    
    return annotated_frame