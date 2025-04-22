import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def detect(input_csv_folder, model, top_features, feature_medians):

    # Function to aggregate features for a single CSV file
    def aggregate_features(csv_path, csv_file):
        data = pd.read_csv(csv_path)
        
        print(f"CSV_PATH CONTENTS ARE: \n {data} \n END VALUES")
        aggregated_features = {
            'video_id': csv_file,
            # Get first column regardless of name
            'frame_ids': data.iloc[:, 0].astype(str).str.cat(sep=',')
        }
        
        aggregated_features['duration_seconds'] = len(data) / 15
        # Arm Bend Status (Normalized)
        aggregated_features['left_arm_bent_proportion'] = (data['Left_arm_status'] == 'BENT').mean()
        aggregated_features['left_arm_partial_bent_proportion'] = (data['Left_arm_status'] == 'PARTIAL_BENT').mean()
        aggregated_features['left_arm_straight_proportion'] = (data['Left_arm_status'] == 'STRAIGHT').mean()
        
        aggregated_features['right_arm_bent_proportion'] = (data['Right_arm_status'] == 'BENT').mean()
        aggregated_features['right_arm_partial_bent_proportion'] = (data['Right_arm_status'] == 'PARTIAL_BENT').mean()
        aggregated_features['right_arm_straight_proportion'] = (data['Right_arm_status'] == 'STRAIGHT').mean()
        
        # Distance Metrics
        aggregated_features['head_to_l_shoulder_mean'] = data['head_to_l_shoulder'].mean()
        aggregated_features['head_to_r_shoulder_mean'] = data['head_to_r_shoulder'].mean()
        aggregated_features['head_to_l_wrist_mean'] = data['head_to_l_wrist'].mean()
        aggregated_features['head_to_r_wrist_mean'] = data['head_to_r_wrist'].mean()
        aggregated_features['shoulder_distance_mean'] = data['shoulder_distance'].mean()
        aggregated_features['l_elbow_to_wrist_mean'] = data['L_elbow_to_wrist'].mean()
        aggregated_features['r_elbow_to_wrist_mean'] = data['R_elbow_to_wrist'].mean()
        
        # Displacement Metrics
        aggregated_features['l_wrist_displacement_total'] = data['L_wrist_displacement'].mean()
        aggregated_features['r_wrist_displacement_total'] = data['R_wrist_displacement'].mean()
        aggregated_features['l_elbow_displacement_total'] = data['L_elbow_displacement'].mean()
        aggregated_features['r_elbow_displacement_total'] = data['R_elbow_displacement'].mean()
        
        # Velocity Metrics
        aggregated_features['l_wrist_velocity_mean'] = data['L_wrist_velocity'].mean()
        aggregated_features['r_wrist_velocity_mean'] = data['R_wrist_velocity'].mean()
        aggregated_features['l_elbow_velocity_mean'] = data['L_elbow_velocity'].mean()
        aggregated_features['r_elbow_velocity_mean'] = data['R_elbow_velocity'].mean()
        
        # Velocity MAX
        aggregated_features['l_wrist_velocity_max'] = data['L_wrist_velocity'].max()
        aggregated_features['r_wrist_velocity_max'] = data['R_wrist_velocity'].max()
        aggregated_features['l_elbow_velocity_max'] = data['L_elbow_velocity'].max()
        aggregated_features['r_elbow_velocity_max'] = data['R_elbow_velocity'].max()
        
        # Acceleration Metrics
        aggregated_features['l_wrist_acceleration_mean'] = data['L_wrist_acceleration'].mean()
        aggregated_features['r_wrist_acceleration_mean'] = data['R_wrist_acceleration'].mean()
        aggregated_features['l_elbow_acceleration_mean'] = data['L_elbow_acceleration'].mean()
        aggregated_features['r_elbow_acceleration_mean'] = data['R_elbow_acceleration'].mean()
        
        aggregated_features['l_wrist_acceleration_std'] = data['L_wrist_acceleration'].std()
        aggregated_features['r_wrist_acceleration_std'] = data['R_wrist_acceleration'].std()
        aggregated_features['l_elbow_acceleration_std'] = data['L_elbow_acceleration'].std()
        aggregated_features['r_elbow_acceleration_std'] = data['R_elbow_acceleration'].std()
        
        # Jerkiness Metrics
        aggregated_features['l_wrist_jerkiness_mean'] = data['L_wrist_velocity_jerkiness'].mean()
        aggregated_features['r_wrist_jerkiness_mean'] = data['R_wrist_velocity_jerkiness'].mean()
        aggregated_features['l_elbow_jerkiness_mean'] = data['L_elbow_velocity_jerkiness'].mean()
        aggregated_features['r_elbow_jerkiness_mean'] = data['R_elbow_velocity_jerkiness'].mean()
        
        aggregated_features['l_wrist_jerkiness_std'] = data['L_wrist_velocity_jerkiness'].std()
        aggregated_features['r_wrist_jerkiness_std'] = data['R_wrist_velocity_jerkiness'].std()
        aggregated_features['l_elbow_jerkiness_std'] = data['L_elbow_velocity_jerkiness'].std()
        aggregated_features['r_elbow_jerkiness_std'] = data['R_elbow_velocity_jerkiness'].std()
        
        # Pose Stability
        aggregated_features['pose_stability_index_mean'] = data['pose_stability_index'].mean()
        aggregated_features['pose_stability_index_std'] = data['pose_stability_index'].std()
        
        # Total Distance Traveled
        aggregated_features['l_wrist_total_distance'] = data['L_wrist_total_distance'].mean()
        aggregated_features['r_wrist_total_distance'] = data['R_wrist_total_distance'].mean()
        aggregated_features['l_elbow_total_distance'] = data['L_elbow_total_distance'].mean()
        aggregated_features['r_elbow_total_distance'] = data['R_elbow_total_distance'].mean()
        
        # Magnitude Metrics
        aggregated_features['l_wrist_magnitude_mean'] = data['L_wrist_magnitude'].mean()
        aggregated_features['r_wrist_magnitude_mean'] = data['R_wrist_magnitude'].mean()
        aggregated_features['l_elbow_magnitude_mean'] = data['L_elbow_magnitude'].mean()
        aggregated_features['r_elbow_magnitude_mean'] = data['R_elbow_magnitude'].mean()
        
        print(f"Aggregate Features are: {aggregate_features}        END VALUES")
        return aggregated_features

    # Get list of CSVs to process
    csv_files = [f for f in os.listdir(input_csv_folder) if f.endswith('.csv')]
    
    # Process each video
    predictions = []
    for csv_file in tqdm(csv_files, desc="Processing files"):
        try:
            # Get features and frame IDs separately
            features = aggregate_features(
                os.path.join(input_csv_folder, csv_file), 
                csv_file
            )
            
            print(f"FEATURES: \n{features} \n END VALUES")
            # Split features from frame IDs
            frame_ids = features.pop('frame_ids')  # Remove from features dict
            feature_df = pd.DataFrame([features])
            
            # Preprocess only numerical features
            feature_df = feature_df.replace(-999, np.nan)
            for col in feature_df.columns:
                if col != 'video_id':
                    feature_df[col] = feature_df[col].fillna(feature_medians[col])
            
            # Make prediction
            X_pred = feature_df[top_features]
            # pred = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0][1]
            
            threshold=0.3
            # Custom threshold classification
            if proba > threshold:
                pred = 1  # Distress detected
            else:
                pred = 0  # No distress
            predictions.append({
                'video_id': features['video_id'],
                'distress_prediction': pred,
                'distress_probability': proba,
                'frame_ids': frame_ids  # Use the separated frame IDs
            })
            
        except Exception as e:
            print(f"\nError processing {csv_file}: {str(e)}")
            continue


    # Save results
    output_folder = r"./data/output/csv_output/phase2_output"
    os.makedirs(output_folder, exist_ok=True)
    result_df = pd.DataFrame(predictions)
    result_path = os.path.join(output_folder, 'detect_results.csv')
    result_df.to_csv(result_path, index=False)
    
    print(f"\nPredictions complete! Results saved to {result_path}")
    
    return result_path