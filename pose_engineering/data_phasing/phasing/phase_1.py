import pandas as pd
import numpy as np
import os

#USES EUCLIDEAN DISTANCE FORMULA para makuha distance.
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#dito yung logic to know if fully bent, partially bent, or straight yung arms. 
def classify_arm_bend(shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y):
    """Calculate the angle between shoulder, elbow, and wrist, and classify the arm bend status."""
    
    threshold1 = 30
    threshold2 = 20
    # Convert inputs to numpy arrays
    shoulder = np.array([shoulder_x, shoulder_y])
    elbow = np.array([elbow_x, elbow_y])
    wrist = np.array([wrist_x, wrist_y])
    
    # Calculate vectors
    upper_arm = shoulder - elbow
    forearm = wrist - elbow
    
    # Calculate angle using the cosine rule
    cosine_angle = np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    # Classify arm bend status
    if angle > 160 - threshold1:  # Straight arm
        status = "STRAIGHT"
    elif angle > 90 + threshold2:  # Partially bent arm
        status = "PARTIAL_BENT"
    elif angle > 1:  # Fully bent arm
        status = "BENT"
    else:
        status = "INVALID"
    
    return status

def phase_1(csv_file_path):
    """
    Phase 1: Extract features from a single CSV file.
    Processes the CSV, calculates various features, and saves results.
    Returns path to the processed CSV file.
    """
    # Check if input file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        return None

    try:
        # Load data with error handling
        data = pd.read_csv(csv_file_path)
        if data.empty:
            print("Error: CSV file is empty")
            return None
            
        # Forward/backward fill missing values
        data = data.ffill().bfill()

        # Create features dataframe
        features_df = pd.DataFrame()
        features_df['frame'] = data.index

        # ANALYZE ARM BENDS
                # ANALYZE ARM BENDS
        features_df['Left_arm_status'] = data.apply(
            lambda row: classify_arm_bend(
                row['L_shoulder_x'], row['L_shoulder_y'],
                row['L_elbow_x'], row['L_elbow_y'],
                row['L_wrist_x'], row['L_wrist_y']
            ),
            axis=1
        )

        features_df['Right_arm_status'] = data.apply(
            lambda row: classify_arm_bend(
                row['R_shoulder_x'], row['R_shoulder_y'],
                row['R_elbow_x'], row['R_elbow_y'],
                row['R_wrist_x'], row['R_wrist_y']
            ),
            axis=1
        )
        # Calculate distances ng specified na keypoints for each frame
        #MORE ON POSTURE NG HEAD tsak SHOULDERS
        features_df['head_to_l_shoulder'] = calculate_distance(data['Head_x'], data['Head_y'], data['L_shoulder_x'], data['L_shoulder_y'])
        features_df['head_to_r_shoulder'] = calculate_distance(data['Head_x'], data['Head_y'], data['R_shoulder_x'], data['R_shoulder_y'])
        features_df['head_to_l_wrist'] = calculate_distance(data['Head_x'], data['Head_y'], data['L_wrist_x'], data['L_wrist_y'])
        features_df['head_to_r_wrist'] = calculate_distance(data['Head_x'], data['Head_y'], data['R_wrist_x'], data['R_wrist_y'])

        # DISPLACEMENT Calculations - distance ng traveled ng keypoint from frame a to frame b
        features_df['L_wrist_displacement'] = np.sqrt((data['L_wrist_x'].diff()**2) + (data['L_wrist_y'].diff()**2)).fillna(0).rolling(window=5).sum()
        features_df['R_wrist_displacement'] = np.sqrt((data['R_wrist_x'].diff()**2) + (data['R_wrist_y'].diff()**2)).fillna(0).rolling(window=5).sum()
        features_df['L_elbow_displacement'] = np.sqrt((data['L_elbow_x'].diff()**2) + (data['L_elbow_y'].diff()**2)).fillna(0).rolling(window=5).sum()
        features_df['R_elbow_displacement'] = np.sqrt((data['R_elbow_x'].diff()**2) + (data['R_elbow_y'].diff()**2)).fillna(0).rolling(window=5).sum()

        #Additional calculations for calculating POSE STABILITY
        features_df['L_elbow_to_wrist'] = calculate_distance(
            data['L_elbow_x'], data['L_elbow_y'],
            data['L_wrist_x'], data['L_wrist_y']
        )

        features_df['R_elbow_to_wrist'] = calculate_distance(
            data['R_elbow_x'], data['R_elbow_y'],
            data['R_wrist_x'], data['R_wrist_y']
        )

        features_df['shoulder_distance'] = calculate_distance(
            data['L_shoulder_x'], data['L_shoulder_y'],
            data['R_shoulder_x'], data['R_shoulder_y']
        )

        # POSE STABILITY INDEX, just sums up all then averages. 
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
        features_df['L_wrist_velocity'] = np.sqrt((data['L_wrist_x'].diff() ** 2) + (data['L_wrist_y'].diff() ** 2)).fillna(0)
        features_df['R_wrist_velocity'] = np.sqrt((data['R_wrist_x'].diff() ** 2) + (data['R_wrist_y'].diff() ** 2)).fillna(0)
        features_df['L_elbow_velocity'] = np.sqrt((data['L_elbow_x'].diff()**2) + (data['L_elbow_y'].diff()**2)).fillna(0)
        features_df['R_elbow_velocity'] = np.sqrt((data['R_elbow_x'].diff()**2) + (data['R_elbow_y'].diff()**2)).fillna(0)

        # ACCELERATION Calculations (original)
        features_df['L_wrist_acceleration'] = features_df['L_wrist_velocity'].diff().fillna(0)
        features_df['R_wrist_acceleration'] = features_df['R_wrist_velocity'].diff().fillna(0)
        features_df['L_elbow_acceleration'] = features_df['L_elbow_velocity'].diff().fillna(0)
        features_df['R_elbow_acceleration'] = features_df['R_elbow_velocity'].diff().fillna(0)

        # FOR IDENTIFYING INTENSITY through velocity lang tsaka acceleration
        for joint in ['L_wrist', 'R_wrist', 'L_elbow', 'R_elbow']:
            features_df[f'{joint}_magnitude'] = np.sqrt(features_df[f'{joint}_velocity']**2 + features_df[f'{joint}_acceleration'].abs()**2)

        #Jerkiness(sudden movements) CALCULATION BASED FROM RAW VELOCITY
        features_df['L_wrist_velocity_jerkiness'] = np.abs(features_df['L_wrist_velocity'] - features_df['L_wrist_velocity'].shift())
        features_df['R_wrist_velocity_jerkiness'] = np.abs(features_df['R_wrist_velocity'] - features_df['R_wrist_velocity'].shift())
        features_df['L_elbow_velocity_jerkiness'] = np.abs(features_df['L_elbow_velocity'] - features_df['L_elbow_velocity'].shift())
        features_df['R_elbow_velocity_jerkiness'] = np.abs(features_df['R_elbow_velocity'] - features_df['R_elbow_velocity'].shift())

        # Calculate total distance traveled for each joint
        features_df['L_wrist_total_distance'] = np.sqrt((data['L_wrist_x'].diff()**2) + (data['L_wrist_y'].diff()**2)).cumsum().fillna(0)
        features_df['R_wrist_total_distance'] = np.sqrt((data['R_wrist_x'].diff()**2) + (data['R_wrist_y'].diff()**2)).cumsum().fillna(0)
        features_df['L_elbow_total_distance'] = np.sqrt((data['L_elbow_x'].diff()**2) + (data['L_elbow_y'].diff()**2)).cumsum().fillna(0)
        features_df['R_elbow_total_distance'] = np.sqrt((data['R_elbow_x'].diff()**2) + (data['R_elbow_y'].diff()**2)).cumsum().fillna(0)


        # Handle remaining NaN values
        features_df.fillna(0, inplace=True)

        # Create output directory
        output_dir = r"./data/output/csv_output/phase1_output"
        os.makedirs(output_dir, exist_ok=True)

        # Create output filename
        input_filename = os.path.basename(csv_file_path)
        output_filename = f"phase1_{input_filename}"
        output_path = os.path.join(output_dir, output_filename)

        # Save results
        features_df.to_csv(output_path, index=False)
        print(f"✅ Processed file saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing file {csv_file_path}: {str(e)}")
        return None