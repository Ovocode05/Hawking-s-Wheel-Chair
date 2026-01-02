import os
import glob
import time
import pandas as pd
import sys
# Redirect stdout/stderr to files
sys.stdout = open('debug_log.txt', 'w')
sys.stderr = open('debug_error.txt', 'w')

from utils import load_data, trim_data, segment_data, calculate_references, process_utterance_segment

def debug_main():
    print("Debug script started.")
    input_root = "Data/recordings"
    csv_files = glob.glob(os.path.join(input_root, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        print("No CSV files found!")
        return

    test_file = csv_files[0]
    print(f"Testing on: {test_file}")
    
    start_time = time.time()
    
    try:
        df = load_data(test_file)
        print(f"Load time: {time.time() - start_time:.2f}s")
        
        df = trim_data(df, limit=1800)
        segments = segment_data(df, frames_per_segment=150)
        print(f"Segment time: {time.time() - start_time:.2f}s")
        
        if len(segments) < 3:
            print("Not enough segments")
            return

        ref_min = segments[0]
        ref_max = segments[1]
        ref_angle_min, ref_angle_max = calculate_references(ref_min, ref_max)
        
        utterances = segments[2:]
        processed_segments = []
        
        print("Starting KNN Imputation...")
        seg_start = time.time()
        for i, seg in enumerate(utterances):
            processed_seg = process_utterance_segment(seg, ref_angle_min, ref_angle_max)
            processed_segments.append(processed_seg)
            print(f"Processed segment {i} in {time.time() - seg_start:.2f}s")
            seg_start = time.time()
            
        print(f"Total processing time: {time.time() - start_time:.2f}s")
        
        final_df = pd.concat(processed_segments, ignore_index=True)
        print(f"Final DF Shape: {final_df.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main()
