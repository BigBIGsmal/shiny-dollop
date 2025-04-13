# utils.py
import csv
import os
import numpy as np

def process_dist_analysis_results(dist_analysis_results):
    """Process distance metrics into a structured format for CSV writing"""
    processed = {
        'left_wrist_distance': dist_analysis_results.get('L_wrist', np.nan),
        'right_wrist_distance': dist_analysis_results.get('R_wrist', np.nan)
    }
    fieldnames = list(processed.keys())
    return processed, fieldnames

def process_bend_analysis_results(bend_analysis_results):
    """Processes Bend analysis results into a flat dictionary for CSV writing."""
    processed = {}
    
    # Process left arm data
    left_data = bend_analysis_results.get('left', {})
    processed.update({
        'left_status': left_data.get('status', np.nan),
        'left_angle': left_data.get('angle', np.nan),
        'left_direction': left_data.get('direction', np.nan),
        'left_count': left_data.get('count', np.nan)
    })
    
    # Process right arm data
    right_data = bend_analysis_results.get('right', {})
    processed.update({
        'right_status': right_data.get('status', np.nan),
        'right_angle': right_data.get('angle', np.nan),
        'right_direction': right_data.get('direction', np.nan),
        'right_count': right_data.get('count', np.nan)
    })
    
    fieldnames = list(processed.keys())
    return processed, fieldnames

def write_behavior_feature_analysis_to_csv(bend_data, dist_data, bend_columns, dist_columns, filename):
    """Write combined bend and distance data to CSV."""
    filename_csv = filename + ".csv"
    save_dir = r"C:\Users\rafae\Documents\Projects\ShinyDollop\pose_engineering\features_csv"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename_csv)
    
    # Combine data and columns
    combined_data = {**bend_data, **dist_data}
    combined_columns = bend_columns + dist_columns
    
    file_exists = os.path.isfile(full_path)
    
    with open(full_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=combined_columns)
        
        if not file_exists:
            writer.writeheader()  # Write header only once
        
        writer.writerow(combined_data)  # Write combined data