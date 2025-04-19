import os
import cv2
import re
import pandas as pd
import numpy as np
from utilities.utils import clear_directory_contents


# Define keypoints and connections for the skeleton
keypoints = ['Head', 'L_shoulder', 'R_shoulder', 'R_elbow', 'R_wrist', 'L_elbow', 'L_wrist']
skeleton_connections = [
    ('Head', 'L_shoulder'), ('Head', 'R_shoulder'),
    ('L_shoulder', 'L_elbow'), ('L_elbow', 'L_wrist'),
    ('R_shoulder', 'R_elbow'), ('R_elbow', 'R_wrist')
]

def phase_0(input_folder, model):
    
    parent_folder = os.path.basename(os.path.dirname(input_folder))
    print(f"Parent folder: {parent_folder}")
    print(f"Input folder: {input_folder}")  
    # Create output folder for annotated frames
    # video_name = os.path.basename(os.path.normpath(input_folder))
    # annotated_frames_folder = os.path.join(output_folder, f"annotated_frames")
    # os.makedirs(annotated_frames_folder, exist_ok=True)
    
    # Prepare CSV data
    csv_data = []
    
    # List and sort all image files in input folder
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    
    # Sort files by numerical value in filename (extracts last number in filename)
    def get_frame_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0
    image_files.sort(key=get_frame_number)
    
    # Process each frame
    for frame_num, image_file in enumerate(image_files):
        frame_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Could not read frame: {image_file}")
            continue
        
        # Process frame with model
        results = model(frame)
        keypoint_data = []
        annotated_frame = frame.copy()
        bounding_box = []
        
        if results:
            for result in results:
                # Extract keypoints
                if result.keypoints is not None:
                    keypoints_np = result.keypoints.xy.cpu().numpy()
                    if keypoints_np.shape[0] > 0:
                        for i, (x, y) in enumerate(keypoints_np[0]):
                            keypoint_data.append((keypoints[i], x, y))
                            # Draw keypoints
                            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                            cv2.putText(annotated_frame, keypoints[i], (int(x) + 5, int(y) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Extract bounding boxes
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f"{conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        bounding_box = [x1, y1, x2, y2]
        
        # Draw skeleton connections
        if keypoint_data:
            for connection in skeleton_connections:
                part1, part2 = connection
                idx1 = keypoints.index(part1)
                idx2 = keypoints.index(part2)
                if idx1 < len(keypoint_data) and idx2 < len(keypoint_data):
                    x1, y1 = keypoint_data[idx1][1], keypoint_data[idx1][2]
                    x2, y2 = keypoint_data[idx2][1], keypoint_data[idx2][2]
                    cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # Save annotated frame
        
        output_dir = os.path.join("data", "temp", parent_folder, "annotated")
        output_frame_path = os.path.join(output_dir, f"frame_{frame_num}.jpg")
        cv2.imwrite(output_frame_path, annotated_frame)
        
        # Prepare CSV row
        frame_info = [frame_num]
        if keypoint_data:
            for key in keypoints:
                found = False
                for kp in keypoint_data:
                    if kp[0] == key:
                        frame_info.extend([kp[1], kp[2]])
                        found = True
                        break
                if not found:
                    frame_info.extend([np.nan, np.nan])
        else:
            frame_info.extend([np.nan] * (len(keypoints) * 2))
        
        # Add bounding box data
        frame_info.extend(bounding_box if bounding_box else [np.nan] * 4)
        
        csv_data.append(frame_info)
    
    # Write CSV
    columns = ['Frame Number']
    for key in keypoints:
        columns.extend([f"{key}_x", f"{key}_y"])
    columns.extend(['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
    phase_0_output_dir = r"./data/output/csv_output/phase0_output"
    
    csv_filename = os.path.join(phase_0_output_dir, f"{parent_folder}_keypoints.csv")
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(csv_filename, index=False)
    print(f"Processing complete. CSV saved to {phase_0_output_dir}")
    
    """Delete the temporary folder after processing"""
    clear_directory_contents(input_folder)
    # clear_directory_contents(output_dir)
    return output_frame_path, csv_filename