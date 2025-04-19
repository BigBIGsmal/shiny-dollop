import streamlit as st
import os
import glob
from utilities.utils import split_vid, save_frames, merge_csv, merge_annotated_frames
import cv2
from data_phasing.phasing.phase_0 import phase_0
from threading import Thread

from ultralytics import YOLO

 # Change if needed


def get_video_files(directory):
    """Get all video files from the specified directory."""
    # Common video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
    
    print(f"Found {video_files} video files in {directory}")
    # Return just the filenames, not the full paths
    return [os.path.basename(file) for file in video_files]

            # Create thread-safe processing with separate models


def run_phase_0(input_dir, results):
    # Each thread gets its own model instance
    local_model = YOLO(r'C:\Users\rafae\Documents\Projects\thesis\dabest.pt')
    annotated_path, csv_path = phase_0(input_dir, local_model)
    results.extend([annotated_path, csv_path])
                
def main():
    st.title("Video Processing App")
    
    # Set the directory path
    video_dir = r"./data/input"
    
    # Check if directory exists
    if not os.path.exists(video_dir):
        st.error(f"Directory not found: {video_dir}")
        st.info("Please make sure the directory exists and contains video files.")
        return
    
    print(f"Directory exists: {video_dir}")
    # Get video files
    video_files = get_video_files(video_dir)
    
    if not video_files:
        st.warning(f"No video files found in {video_dir}")
        return
    
    # Create a dropdown to select a video
    selected_video = st.selectbox(
        "Select a video to process:",
        options=video_files
    )
    
    # Display the selected video
    if selected_video:
        print(f"Selected video: {selected_video}")
        video_path = os.path.join(video_dir, selected_video)
        
        print(f"Video path: {video_path}")
        st.video(video_path)
        
        # Add a button to process the video
        if st.button("Process Video"):
            st.info(f"Processing video: {video_path}")
            split_1, split_2 = split_vid(video_path)
            print(f"Split 1: {split_1} frames END VALUE")
            
            temp_dir = r"./data/temp"
            split_1_dir = os.path.join(temp_dir, "split_1/unannotated")
            split_2_dir = os.path.join(temp_dir, "split_2/unannotated")

            # Create directories if they don't exist
            os.makedirs(split_1_dir, exist_ok=True)
            os.makedirs(split_2_dir, exist_ok=True)
            
            save_frames(split_1, split_1_dir)
            
            save_frames(split_2, split_2_dir)
            
            """ 
            We are replacing the unannotated frames with the annotated frames
            in the temporary folder (temp).
            """
            # Create containers for thread results
            results1, results2 = [], []
                        # Create and start threads
            t1 = Thread(target=run_phase_0, args=(split_1_dir, results1))
            t2 = Thread(target=run_phase_0, args=(split_2_dir, results2))
            
            t1.start()
            t2.start()
            
            # Wait for threads to complete
            t1.join()
            t2.join()
            
            # Unpack results
            out_a, out_b = results1
            output_a2, output_b2 = results2
            print(f"FIRST THREAD Annotated frames saved to: {out_a}")
            print(f"FIRST THREAD CSV file saved to: {out_b}")
            
            print(f"SECOND THREAD Annotated frames saved to: {output_a2}")
            print(f"SECOND THREAD CSV file saved to: {output_b2}")
            
            phase_0_csv = merge_csv(out_b, output_b2)
            print(f"CSV files merged successfully: {out_b} + {output_b2}")
            print(f"{phase_0_csv} frame_IDs refractored")
            
            phase_0_frames = merge_annotated_frames(out_a, output_a2)
            print(f"Annotated frames merged successfully: {out_a} + {output_a2}")
            print(f"Saved to : {phase_0_frames} End of phase 0")
            # Here you would add your video processing code
            # For example:
            # process_video(video_path)
            st.success("Video processing complete!")

if __name__ == "__main__":
    main()