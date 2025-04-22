import cv2
import os
import shutil
import pandas as pd
import re
import subprocess
from pydub import AudioSegment
from pydub.generators import Sine
import math


def clear_temp_dir(temp_dir):
    """Clears the temporary directory."""
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Temporary directory does not exist: {temp_dir}")
def save_frames(frames, output_dir):
    """Saves frames to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
def split_vid(selected_vid):
    """Splits the video into two parts based on the number of frames."""
    # Check if file exists first
    if not os.path.exists(selected_vid):
        print(f"Error: File not found at {os.path.abspath(selected_vid)}")
        return None, None

    cap = cv2.VideoCapture(selected_vid)
    if not cap.isOpened():
        # Try to get more detailed error information
        print(f"Failed to open video: {selected_vid}")
        print("Possible reasons:")
        print("- File path is incorrect")
        print("- Unsupported video format")
        print("- Missing codecs")
        print("- Corrupted video file")
        print(f"Absolute path: {os.path.abspath(selected_vid)}")
        return None, None

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: Video contains 0 frames")
        cap.release()
        return None, None

    split_point = total_frames // 2
    split_1, split_2 = [], []

    # Read first half
    for _ in range(split_point):
        ret, frame = cap.read()
        if not ret:
            print(f"Stopped early at frame {len(split_1)}")
            break
        split_1.append(frame)

    # Read second half
    for _ in range(split_point, total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Stopped early at frame {split_point + len(split_2)}")
            break
        split_2.append(frame)

    cap.release()
    
    print(f"Split successful: {len(split_1)} + {len(split_2)} = {len(split_1)+len(split_2)} frames")
    return split_1, split_2
# Add this after processing in main.py
def clear_directory_contents(directory):
    """Remove all files and subdirectories within a directory, but keep the directory itself"""
    for entry in os.scandir(directory):
        try:
            if entry.is_file():
                os.unlink(entry.path)  # Delete file
            else:
                shutil.rmtree(entry.path)  # Delete subdirectory
        except Exception as e:
            print(f"Error deleting {entry.path}: {e}")
            
def merge_csv(split_1_csv, split_2_csv):
    
    """Merges two CSV files into one."""
    if not os.path.exists(split_1_csv):
        print(f"Error: File not found at {os.path.abspath(split_1_csv)}")
        return None

    if not os.path.exists(split_2_csv):
        print(f"Error: File not found at {os.path.abspath(split_2_csv)}")
        return None

    # Read the first CSV file
    df1 = pd.read_csv(split_1_csv)
    
    # Read the second CSV file
    df2 = pd.read_csv(split_2_csv)

    # Concatenate the two DataFrames
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_csv_path = os.path.join(os.path.dirname(split_1_csv), "merged.csv")
    merged_df.to_csv(merged_csv_path, index=False)
    
    print(f"Merged CSV saved to: {merged_csv_path}")
    
    """delete   the two csv files"""
    os.remove(split_1_csv)
    os.remove(split_2_csv) 
    phase_0_csv = assign_frame_id(merged_csv_path)
    
    return phase_0_csv

def merge_annotated_frames(output_a, output_a2):
    """
    Merges two frame directories while maintaining original split order and
    renumbers frames sequentially from 0. Returns path to merged directory.
    """ 
    merged_dir = r"./data/output/frame_output"
    
    def process_split(split_path, start_idx):
        """Process a single split directory and return next available index"""
        # Get directory from file path
        split_dir = os.path.dirname(split_path)
        
        # Collect and sort frames
        frames = []
        for f in os.listdir(split_dir):
            match = re.match(r"frame_(\d+)\.(jpg|png)", f)
            if match:
                frame_num = int(match.group(1))
                frames.append((frame_num, f))
        
        # Sort by original frame number
        frames.sort(key=lambda x: x[0])
        
        # Copy with new numbering
        for idx, (orig_num, fname) in enumerate(frames, start=start_idx):
            src = os.path.join(split_dir, fname)
            dst = os.path.join(merged_dir, f"frame_{idx}.jpg")
            shutil.copy2(src, dst)
        
        return len(frames) + start_idx

    # Process first split starting at 0
    next_idx = process_split(output_a, 0)
    
    # Process second split continuing from last index
    process_split(output_a2, next_idx)
    
    print(f"Merged frames saved to: {merged_dir}")
    split_1 = r"./data/temp/split_1/annotated"
    split_2 = r"./data/temp/split_2/annotated"
    clear_directory_contents(split_1)
    clear_directory_contents(split_2)
    
    return merged_dir

def split_csv_phase1():
    csv_path = r"./data/output/csv_output/phase1_output"
    split_output = r"./data/temp/phase_1_splits"
    os.makedirs(split_output, exist_ok=True)
    
    # Get all CSV files in phase1_output
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        # Read original CSV
        full_path = os.path.join(csv_path, csv_file)
        df = pd.read_csv(full_path)
        
        # Calculate split parameters (8 seconds = 240 frames @30fps)
        total_frames = len(df)
        frames_per_split = 10
        num_splits = math.ceil(total_frames / frames_per_split)
        
        # Split the DataFrame
        splits = []
        for i in range(num_splits):
            start_idx = i * frames_per_split
            end_idx = (i+1) * frames_per_split
            split_df = df.iloc[start_idx:end_idx]
            splits.append(split_df)
        
        # Save splits
        split_files = []
        for i, split_df in enumerate(splits, 1):
            split_name = f"{os.path.splitext(csv_file)[0]}_split_{i}.csv"
            split_path = os.path.join(split_output, split_name)
            split_df.to_csv(split_path, index=False)
            split_files.append(split_path)
        

    
    return split_output


def create_video_from_frames():
    video_path = r"./data/output/video_output"
    labels_path = r"./data/output/csv_output/phase2_output/detect_results.csv"
    frames_path = r"./data/output/frame_output"
    video_temp = r"./data/temp/video_temp"
    
    # Create output directory
    os.makedirs(video_path, exist_ok=True)
    
    # Create output directory
    os.makedirs(video_temp, exist_ok=True)
    
    # Load labels data
    labels_df = pd.read_csv(labels_path)
    
    # Prepare frame lists
    frame_details = []
    
    # Process each row in labels CSV
    for _, row in labels_df.iterrows():
        frames = list(map(int, row['frame_ids'].split(',')))
        for frame_id in frames:
            frame_details.append({
                'frame_id': frame_id,
                'prediction': row['distress_prediction'],
                'probability': row['distress_probability']
            })
    
    # Sort frames by ID
    frame_details.sort(key=lambda x: x['frame_id'])
    
    # Initialize video parameters
    fps = 30
    temp_video_path = os.path.join(video_temp, "temp_vid.mp4")
    final_video_path = os.path.join(video_path, "final_output.mp4")
    
    # Initialize distress tracking
    distress_segments = []
    current_distress_start = None
    
    # Create video writer
    sample_frame = cv2.imread(os.path.join(frames_path, f"frame_{frame_details[0]['frame_id']}.jpg"))
    height, width, _ = sample_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Process frames and track distress periods
    for idx, detail in enumerate(frame_details):
        frame_file = os.path.join(frames_path, f"frame_{detail['frame_id']}.jpg")
        if not os.path.exists(frame_file):
            continue

        frame = cv2.imread(frame_file)
        
        # Add annotation
        label = "DISTRESSED" if detail['prediction'] == 1 else "NORMAL"
        color = (0, 0, 255) if detail['prediction'] == 1 else (0, 255, 0)
        
        cv2.putText(frame, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Prob: {detail['probability']:.2f}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        video_writer.write(frame)
        
        # Track distress periods for audio
        if detail['prediction'] == 1:
            if current_distress_start is None:
                current_distress_start = idx / fps
        else:
            if current_distress_start is not None:
                distress_segments.append((
                    current_distress_start,
                    (idx - 1) / fps
                ))
                current_distress_start = None
                
    # Add final distress segment if needed
    if current_distress_start is not None:
        distress_segments.append((
            current_distress_start,
            (len(frame_details) - 1) / fps
        ))

    video_writer.release()

    # Create audio with distress beeps
    total_duration = len(frame_details) / fps
    audio = AudioSegment.silent(duration=total_duration * 1000)  # Convert to milliseconds

    # Generate beeps for distress periods
    for start, end in distress_segments:
        duration = end - start
        beep = Sine(1000).to_audio_segment(duration=duration * 1000 - 50).append(
            AudioSegment.silent(50), crossfade=50)
        audio = audio.overlay(beep, position=start * 1000)

    # Save temporary audio
    temp_audio_path = os.path.join(video_temp, "temp_audio.wav")
    audio.export(temp_audio_path, format="wav")
    print(f"Temporary audio saved to: {temp_audio_path}")
    print(f"Temporary video saved to: {temp_video_path}")
    # Combine video and audio using ffmpeg
    subprocess.run([
        'ffmpeg', '-y',
        '-i', temp_video_path,
        '-i', temp_audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        final_video_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"{final_video_path}")
    # Cleanup temporary files
    os.remove(temp_video_path)
    os.remove(temp_audio_path)

    print(f"Final video with distress alerts created at: {final_video_path}")
    return final_video_path

def assign_frame_id(csv_file):
    """Reassigns sequential frame numbers starting from 0 in the first column of the CSV."""
    try:
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print("Error: CSV file is empty")
            return None
        
        # Get the first column name dynamically
        frame_column = df.columns[0]
        
        # Reset numbering starting from 0
        df[frame_column] = range(0, len(df))
        
        # Save updated CSV
        df.to_csv(csv_file, index=False)
        print(f"Frame numbers reset to 0-index in: {csv_file}")
        return csv_file
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        return None