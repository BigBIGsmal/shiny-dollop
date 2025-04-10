import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from keypoint.extract import get_keypoint_coordinates, draw_skeleton, display_keypoint_info

    # Constants
KEYPOINTS = ['Head', 'L_shoulder', 'R_shoulder', 'R_elbow', 'R_wrist', 'L_elbow', 'L_wrist']
SKELETON_CONNECTIONS = [
    ('Head', 'L_shoulder'), ('Head', 'R_shoulder'),
    ('L_shoulder', 'L_elbow'), ('L_elbow', 'L_wrist'),
    ('R_shoulder', 'R_elbow'), ('R_elbow', 'R_wrist')
]


# Initialize model
model = YOLO(r'C:\Users\rafae\Documents\Projects\thesis\dabest.pt')

def process_video_frame(frame, frame_num, frame_width, frame_height):
    """
    Process a single video frame through the pipeline
    Returns:
        - annotated_frame: Frame with visualizations
        - keypoint_data: List of (keypoint_name, x, y) tuples
        - bbox: Bounding box coordinates [x1, y1, x2, y2]
    """
    results = model(frame)
    keypoint_data, bbox = get_keypoint_coordinates(results, frame_width, frame_height, KEYPOINTS)
    annotated_frame = draw_skeleton(frame, keypoint_data, bbox, SKELETON_CONNECTIONS)
    annotated_frame = display_keypoint_info(annotated_frame, keypoint_data, frame_num)
    
    return annotated_frame, keypoint_data, bbox

def extract_frames(input_video, output_folder, fps_target=15):
    """Main video processing function"""
    cap = cv2.VideoCapture(input_video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    frame_output_dir = os.path.join(output_folder, f"{video_name}_frames")
    os.makedirs(frame_output_dir, exist_ok=True)
    
    # CSV setup
    csv_data = []
    columns = ['frame_num'] + [f"{kp}_x" for kp in KEYPOINTS] + [f"{kp}_y" for kp in KEYPOINTS] + \
              ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
    
    frame_num = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % int(original_fps / fps_target) == 0:
            # Process frame
            annotated_frame, keypoint_data, bbox = process_video_frame(
                frame, frame_num, width, height)
            
            # Save frame
            cv2.imwrite(os.path.join(frame_output_dir, f"frame_{frame_num:04d}.jpg"), annotated_frame)
            cv2.imshow('Pose Estimation', annotated_frame)
            
            # Store data for CSV
            frame_info = [frame_num]
            if keypoint_data:
                kp_dict = {k: (x, y) for k, x, y in keypoint_data}
                for kp in KEYPOINTS:
                    if kp in kp_dict:
                        frame_info.extend(kp_dict[kp])
                    else:
                        frame_info.extend([np.nan, np.nan])
            else:
                frame_info.extend([np.nan] * len(KEYPOINTS) * 2)
            
            frame_info.extend(bbox if len(bbox) == 4 else [np.nan]*4)
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
    output_dir = r"C:\Users\rafae\Documents\Projects\thesis\pose_engineering\tests"
    process_all_videos(input_dir, output_dir)