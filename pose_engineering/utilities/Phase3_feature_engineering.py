import os
import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bar

def phase_3(csv_folder, output_path):
    """
    Process all Phase 1 CSV files into aggregated video-level features
    :param csv_folder: Path to folder containing Phase 1 CSVs
    :param output_path: Path to save final dataset
    """
    # Get all CSV files
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    
    print(f"\nFound {csv_files} CSV files in {csv_folder}.")
    # Initialize storage
    video_features = []
    
    # Process files with progress bar
    for csv_file in tqdm(csv_files, desc="Processing videos"):
        try:
            # Read CSV with error handling
            df = pd.read_csv(os.path.join(csv_folder, csv_file))
            
            # Create video entry
            label = 1 if 'distress' in csv_file.lower() else 0
            if 'notdistress' in csv_file.lower():
                label = 0

            entry = {
                'video_id': os.path.splitext(csv_file)[0],
                'label': label
            }
            
            # Calculate aggregates for each feature
            for col in df.columns:
                if col == 'frame':
                    continue
                
                # Basic statistics
                entry[f'{col}_mean'] = df[col].mean()
                entry[f'{col}_std'] = df[col].std()
                entry[f'{col}_max'] = df[col].max()
                entry[f'{col}_min'] = df[col].min()
                entry[f'{col}_median'] = df[col].median()
                
                # Advanced temporal features
                entry[f'{col}_range'] = entry[f'{col}_max'] - entry[f'{col}_min']
                entry[f'{col}_q95'] = df[col].quantile(0.95)
                entry[f'{col}_skew'] = df[col].skew()
                
                # Slope (linear trend)
                if len(df) > 1:
                    entry[f'{col}_slope'] = np.polyfit(np.arange(len(df)), df[col], 1)[0]
                else:
                    entry[f'{col}_slope'] = 0
            
            video_features.append(entry)
            
        except Exception as e:
            print(f"\nError processing {csv_file}: {str(e)}")
            continue
    
    # Create DataFrame and save
    final_df = pd.DataFrame(video_features)
    
    # Handle missing values
    final_df.fillna(-999, inplace=True)  # XGBoost-friendly missing value handling
    
    print(f"Tracking output path in phase 3: {output_path} END VALUES")
    # Save output
    final_df.to_csv(f"{output_path}_phase3_.csv", index=False)

    print(f"\nSuccessfully processed {len(final_df)} videos. Saved to {output_path}")