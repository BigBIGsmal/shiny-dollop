# main.py (updated)

import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from keypoint.extract import get_keypoint_coordinates, draw_skeleton, display_keypoint_info
from behave.bend import analyze_bends, draw_bend_visuals
from behave.distance import analyze_dist

# metric feature analysis is unstable
# from behave.metrics import init_feature_state, calculate_pose_stability, update_temporal_features, draw_features, calculate_distance
from utilities.csvkit import process_behavior_analysis_results, write_behavior_feature_analysis_to_csv, create_phase2_csv



# Constants
KEYPOINTS = ['Head', 'L_shoulder', 'R_shoulder', 'R_elbow', 'R_wrist', 'L_elbow', 'L_wrist']
SKELETON_CONNECTIONS = [
    ('Head', 'L_shoulder'), ('Head', 'R_shoulder'),
    ('L_shoulder', 'L_elbow'), ('L_elbow', 'L_wrist'),
    ('R_shoulder', 'R_elbow'), ('R_elbow', 'R_wrist')
]
METRIC_JOINTS = ['L_wrist', 'R_wrist', 'L_elbow', 'R_elbow']

# Initialize model
model = YOLO(r'C:\Users\rafae\Documents\Projects\thesis\dabest.pt')

def init_counters():
    """Initialize bend tracking counters"""
    return {
        'left_count': 0,
        'right_count': 0,
        'prev_left_status': None,
        'prev_right_status': None
    }

"""def process_video_frame(frame, frame_num, frame_width, frame_height, counters, feature_state, delta_time, video_name):"""
def process_video_frame(frame, frame_num, frame_width, frame_height, counters, delta_time, video_name):
    """
    Process a single video frame through the pipeline
    Returns:
        - annotated_frame: Frame with visualizations
        - keypoint_data: List of (keypoint_name, x, y) tuples
        - bbox: Bounding box coordinates [x1, y1, x2, y2]
        - analysis_results: Arm behavior analysis data
        - updated counters
    """
    # Let the model process the frame and predict the keypoints
    results = model(frame)
    
    # Get keypoint coordinates and bounding box
    keypoint_data, bbox = get_keypoint_coordinates(results, frame_width, frame_height, KEYPOINTS)
    print(f"Keypoint_data: {keypoint_data} END VALUES")
    
    # Analyze arm behavior
    analysis_results, counters, vis_elements = analyze_bends(keypoint_data, counters)
    
    print(f"\nBend Values Results: {analysis_results} END VALUES")
    # Processes the output targets of the bend analysis and prepares it for CSV writing
    # Target: [left_status, left_angle, left_direction, left_count, right_status, right_angle, right_direction, right_count]

    # write_bend_analysis_to_csv(bend_data, column_names, video_name) for tweaking
    
    # Show skeleton connections to the frame
    annotated_frame = draw_skeleton(frame, keypoint_data, bbox, SKELETON_CONNECTIONS)
    
    # Add the bend visuals to the frame 
    annotated_frame = draw_bend_visuals(annotated_frame, vis_elements, analysis_results, counters)
    
    # Save the frame state with keypoint info and Bend Visuals
    # ang need ko ireturn is ung mismong cv2 elements, hindi ung frame na may annotation
    annotated_frame = display_keypoint_info(annotated_frame, keypoint_data, frame_num)
    
    current_position , acceleration_ave, annotated_frame, total_distance_data= analyze_dist(annotated_frame, keypoint_data)
    
    print(f"keypoints: {keypoint_data} \
            Bounding box: {bbox} \
            Analysis results: {analysis_results} \
                ")
    all_data, all_columns = process_behavior_analysis_results(keypoint_data, analysis_results, total_distance_data, acceleration_ave)
    write_behavior_feature_analysis_to_csv(all_data, all_columns, video_name)
    return annotated_frame, keypoint_data, bbox, analysis_results, counters


def extract_frames(input_video, output_folder, fps_target=15):
    """Main video processing function"""
    cap = cv2.VideoCapture(input_video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    # frame_output_dir = os.path.join(output_folder, f"{video_name}_frames")
    # os.makedirs(frame_output_dir, exist_ok=True)
    
    # CSV setup - now includes behavior data
    csv_data = []
    columns = ['frame_num'] + [f"{kp}_x" for kp in KEYPOINTS] + [f"{kp}_y" for kp in KEYPOINTS] + \
              ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'] + \
              ['left_status', 'left_angle', 'left_direction', 'left_count',
               'right_status', 'right_angle', 'right_direction', 'right_count']
    
    frame_num = 0
    frame_count = 0
    counters = init_counters()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # feature state is for metrics analysis, not used in this version
        # feature_state = init_feature_state()
        delta_time = 1 / original_fps

        if frame_count % int(original_fps / fps_target) == 0:
            # Process frame
            # Removing feature_state for now, since it's not used in the current version
            """annotated_frame, keypoint_data, bbox, analysis_results, counters = process_video_frame(
                frame, frame_num, width, height, counters, feature_state, delta_time, video_name)"""
                
                
            annotated_frame, keypoint_data, bbox, analysis_results, counters = process_video_frame( frame, frame_num, width, height, counters, delta_time, video_name)
            
            
            #kahit ipasa ko nlng dito ung mga need ko ilagay sa frame para sabay sabay sila mag lagay ng drawing sa frame
            # di ko na din need mag lagay ng animations sa frame mismo sa process_video_frame function tsaka sa functions ng behavior mismo
            # kahit dito ko nlng tlaga siya mismo ilagay
            # Save frame
            # cv2.imwrite(os.path.join(frame_output_dir, f"frame_{frame_num:04d}.jpg"), annotated_frame)
            cv2.imshow('Pose Estimation', annotated_frame)
            
            # Store data for CSV
            frame_info = [frame_num]
            
            # Keypoint coordinates
            if keypoint_data:
                kp_dict = {k: (x, y) for k, x, y in keypoint_data}
                for kp in KEYPOINTS:
                    if kp in kp_dict:
                        frame_info.extend(kp_dict[kp])
                    else:
                        frame_info.extend([np.nan, np.nan])
            else:
                frame_info.extend([np.nan] * len(KEYPOINTS) * 2)
            
            # Bounding box
            frame_info.extend(bbox if len(bbox) == 4 else [np.nan]*4)
            
            # Behavior analysis data
            frame_info.extend([
                analysis_results.get('left', {}).get('status', np.nan),
                analysis_results.get('left', {}).get('angle', np.nan),
                analysis_results.get('left', {}).get('direction', np.nan),
                analysis_results.get('left', {}).get('count', np.nan),
                analysis_results.get('right', {}).get('status', np.nan),
                analysis_results.get('right', {}).get('angle', np.nan),
                analysis_results.get('right', {}).get('direction', np.nan),
                analysis_results.get('right', {}).get('count', np.nan)
            ])
            
            csv_data.append(frame_info)
            frame_num += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    # Save CSV
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(os.path.join(output_folder, f"{video_name}_keypoints.csv"), index=False)
    
    cap.release()
    cv2.destroyAllWindows()

def process_all_videos(input_folder, output_folder):
    """Batch process all videos in a folder"""
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            print(f"Processing {filename}...")
            extract_frames(
                os.path.join(input_folder, filename),
                output_folder
            )
            print(f"Completed {filename}")

# Example usage
if __name__ == "__main__":
    input_dir = r"C:\Users\rafae\Documents\Projects\thesis\sample vid\front"
    output_dir = r"C:\Users\rafae\Documents\Projects\ShinyDollop\pose_engineering\data\phase1_output"
    process_all_videos(input_dir, output_dir)
    # create_phase2_csv()