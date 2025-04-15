import csv
import os
import numpy as np
from utilities.Phase2_feature_engineering import extract_video_features 

def process_behavior_analysis_results(raw_keypoints, bend_analysis_results, dist_analysis_results, acceleration_ave):
    """Process distance and acceleration metrics for CSV writing"""
    
    # Convert raw keypoints list to dictionary {joint: (x, y)}
    keypoint_dict = {k: (x, y) for k, x, y in raw_keypoints} if isinstance(raw_keypoints, list) else raw_keypoints
    
    # Helper function to safely extract coordinates
    def get_coord(joint, index):
        coords = keypoint_dict.get(joint, (np.nan, np.nan))
        return coords[index] if len(coords) > index else np.nan
    
    processed = {
        'left_wrist_distance': dist_analysis_results.get('L_wrist', np.nan),
        'right_wrist_distance': dist_analysis_results.get('R_wrist', np.nan),
        'left_wrist_acceleration': acceleration_ave.get('L_wrist', np.nan),
        'right_wrist_acceleration': acceleration_ave.get('R_wrist', np.nan),
        
        # Keypoint coordinates with safe access
        'left_wrist_x': get_coord('L_wrist', 0),
        'left_wrist_y': get_coord('L_wrist', 1),
        'right_wrist_x': get_coord('R_wrist', 0),
        'right_wrist_y': get_coord('R_wrist', 1),
        'left_elbow_x': get_coord('L_elbow', 0),
        'left_elbow_y': get_coord('L_elbow', 1),
        'right_elbow_x': get_coord('R_elbow', 0),
        'right_elbow_y': get_coord('R_elbow', 1),
        'left_shoulder_x': get_coord('L_shoulder', 0),
        'left_shoulder_y': get_coord('L_shoulder', 1),
        'right_shoulder_x': get_coord('R_shoulder', 0),
        'right_shoulder_y': get_coord('R_shoulder', 1),
        'head_x': get_coord('Head', 0),
        'head_y': get_coord('Head', 1),
    }
    
    # Add bend analysis results (may contain strings)
    left_data = bend_analysis_results.get('left', {})
    processed.update({
        'left_status': left_data.get('status', None),  # Non-numeric
        'left_angle': left_data.get('angle', np.nan),
        'left_direction': left_data.get('direction', None),  # Non-numeric
        'left_count': left_data.get('count', np.nan)
    })
    
    right_data = bend_analysis_results.get('right', {})
    processed.update({
        'right_status': right_data.get('status', None),  # Non-numeric
        'right_angle': right_data.get('angle', np.nan),
        'right_direction': right_data.get('direction', None),  # Non-numeric
        'right_count': right_data.get('count', np.nan)
    })
    
    fieldnames = list(processed.keys())
    return processed, fieldnames

def write_behavior_feature_analysis_to_csv(data, column_names, filename):
    """Write combined bend and distance data to CSV."""
    filename_csv = filename + ".csv"
    
    #Change where to save PHASE 1 CSV FEATURES 
    save_dir = r"C:\Users\rafae\Documents\Projects\ShinyDollop\pose_engineering\data\phase1_output"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename_csv)
    
    file_exists = os.path.isfile(full_path)
    
    with open(full_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_names)
        
        if not file_exists:
            writer.writeheader()  # Write header only once
        
        writer.writerow(data)  # Write combined data

        
def create_phase2_csv():
    # Change where you PHASE 1 OUTPUT can be Accessed
    CSV_FOLDER = r'C:\Users\rafae\Documents\Projects\ShinyDollop\pose_engineering\data\phase1_output'  # Update this path
    
    # Change where to save PHASE 2 CSV FEATURES
    OUTPUT_PATH = r"C:\Users\rafae\Documents\Projects\ShinyDollop\pose_engineering\data\phase2_output\csv_f_engineering.csv"


    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    
    extract_video_features(CSV_FOLDER, OUTPUT_PATH)