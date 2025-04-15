import pandas as pd
import numpy as np
import os
from tkinter import filedialog, Tk

# Add error handling for zero vectors
def calculate_angle(x1, y1, x2, y2):
    dot_product = x1 * x2 + y1 * y2
    magnitude1 = np.sqrt(x1**2 + y1**2)
    magnitude2 = np.sqrt(x2**2 + y2**2)
    
    # Handle division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return np.nan
    
    cos_theta = dot_product / (magnitude1 * magnitude2)
    # Handle numerical precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))  # Convert to degrees

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)

Tk().withdraw()

folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        csv_path = os.path.join(folder_path, filename)
        data = pd.read_csv(csv_path)
        data = data.ffill().bfill()

        features_df = pd.DataFrame()
        features_df['frame'] = data.index

        # Calculate distances for each frame
        features_df['head_to_l_shoulder'] = calculate_distance(data['Head_x'], data['Head_y'], data['L_shoulder_x'], data['L_shoulder_y'])
        features_df['head_to_r_shoulder'] = calculate_distance(data['Head_x'], data['Head_y'], data['R_shoulder_x'], data['R_shoulder_y'])
        features_df['head_to_l_wrist'] = calculate_distance(data['Head_x'], data['Head_y'], data['L_wrist_x'], data['L_wrist_y'])
        features_df['head_to_r_wrist'] = calculate_distance(data['Head_x'], data['Head_y'], data['R_wrist_x'], data['R_wrist_y'])

        # DISPLACEMENT Calculations
        features_df['L_wrist_displacement'] = np.sqrt((data['L_wrist_x'].diff()**2) + (data['L_wrist_y'].diff()**2)).fillna(0).rolling(window=5).sum()
        features_df['R_wrist_displacement'] = np.sqrt((data['R_wrist_x'].diff()**2) + (data['R_wrist_y'].diff()**2)).fillna(0).rolling(window=5).sum()
        features_df['L_elbow_displacement'] = np.sqrt((data['L_elbow_x'].diff()**2) + (data['L_elbow_y'].diff()**2)).fillna(0).rolling(window=5).sum()
        features_df['R_elbow_displacement'] = np.sqrt((data['R_elbow_x'].diff()**2) + (data['R_elbow_y'].diff()**2)).fillna(0).rolling(window=5).sum()


        # New comprehensive stability index (average of 7 distances)
        features_df['pose_stability_index'] = (
            features_df['head_to_l_shoulder'] +
            features_df['head_to_r_shoulder'] +
            features_df['head_to_l_wrist'] +
            features_df['head_to_r_wrist'] +
            features_df['L_elbow_to_wrist'] +
            features_df['R_elbow_to_wrist'] +
            features_df['shoulder_distance']
        ) / 7

        # VELOCITY Calculations (original)
        features_df['L_wrist_velocity'] = np.sqrt((data['L_wrist_x'].diff() * 2) + (data['L_wrist_y'].diff() * 2)).fillna(0)
        features_df['R_wrist_velocity'] = np.sqrt((data['R_wrist_x'].diff() * 2) + (data['R_wrist_y'].diff() * 2)).fillna(0)
        features_df['L_elbow_velocity'] = np.sqrt((data['L_elbow_x'].diff()**2) + (data['L_elbow_y'].diff()**2)).fillna(0)
        features_df['R_elbow_velocity'] = np.sqrt((data['R_elbow_x'].diff()**2) + (data['R_elbow_y'].diff()**2)).fillna(0)

        # ACCELERATION Calculations (original)
        features_df['L_wrist_acceleration'] = features_df['L_wrist_velocity'].diff().fillna(0)
        features_df['R_wrist_acceleration'] = features_df['R_wrist_velocity'].diff().fillna(0)
        features_df['L_elbow_acceleration'] = features_df['L_elbow_velocity'].diff().fillna(0)
        features_df['R_elbow_acceleration'] = features_df['R_elbow_velocity'].diff().fillna(0)

        # ======================
        # SMOOTHED VERSIONS (NEW CODE)
        # ======================
        smoothing_window = 5

        for joint in ['L_wrist', 'R_wrist', 'L_elbow', 'R_elbow']:
            # Smoothed velocity
            features_df[f'{joint}_vel_smoothed'] = (
                features_df[f'{joint}_velocity']
                .rolling(window=smoothing_window, min_periods=1)
                .mean()
            )
            
            # Smoothed acceleration
            features_df[f'{joint}_accel_smoothed'] = (
                features_df[f'{joint}_acceleration']
                .rolling(window=smoothing_window, min_periods=1)
                .mean()
            )

        # 5. IMPROVED MAGNITUDE CALCULATIONS
        # ======================

        # New magnitude calculations combining velocity and acceleration
        for joint in ['L_wrist', 'R_wrist', 'L_elbow', 'R_elbow']:
            features_df[f'{joint}_magnitude'] = np.sqrt(
                features_df[f'{joint}_velocity']**2 + 
                features_df[f'{joint}_acceleration'].abs()**2
            )

        # Apply function vectorized
        features_df['shoulder_angle'] = np.vectorize(calculate_angle)(
            data['L_shoulder_x'] - data['Head_x'],
            data['L_shoulder_y'] - data['Head_y'],
            data['R_shoulder_x'] - data['Head_x'],
            data['R_shoulder_y'] - data['Head_y']
        )

        # 1. Bilateral symmetry features
        features_df['wrist_speed_asymmetry'] = np.abs(features_df['L_wrist_velocity'] - features_df['R_wrist_velocity'])

        # 2. Body normalization
        shoulder_width = calculate_distance(data['L_shoulder_x'], data['L_shoulder_y'],
                                        data['R_shoulder_x'], data['R_shoulder_y'])
        features_df['normalized_head_height'] = (data['Head_y'] - (data['L_shoulder_y'] + data['R_shoulder_y'])/2) / shoulder_width

        # 3. Hand proximity features
        features_df['wrist_distance'] = calculate_distance(data['L_wrist_x'], data['L_wrist_y'],data['R_wrist_x'], data['R_wrist_y'])

        # After calculating distances, velocities, and accelerations
        features_df.fillna(0, inplace=True)  # Replace NaN values with 0

        # Output filename
        output_filename = f"extracted_{filename}"
        output_path = os.path.join(folder_path, output_filename)
        features_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

print("All Files processed.")