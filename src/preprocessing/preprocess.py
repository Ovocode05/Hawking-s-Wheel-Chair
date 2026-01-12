import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_data, trim_data, segment_data, process_utterance_segment

def main():
    # Setup paths
    # Determine project root based on script location (src/preprocessing/preprocess.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    input_root = os.path.join(project_root, "Data", "recordings")
    output_root = os.path.join(project_root, "Normalized_dataset", "recordings")
    graph_root = os.path.join(project_root, "Normalized_dataset", "Graphs")
    
    # Ensure output directories exist
    # Logic: We need to replicate subdirectory structure {word}
    
    print(f"Searching for CSV files in {input_root}...")
    # recursive glob for all CSVs
    csv_files = glob.glob(os.path.join(input_root, "**", "*.csv"), recursive=True)
    
    print(f"Found {len(csv_files)} files to process.")
    
    success_count = 0
    error_count = 0
    
    for file_path in csv_files:
        try:
            # Determine relative path to maintain structure
            rel_path = os.path.relpath(file_path, input_root)
            word_dir = os.path.dirname(rel_path)
            file_name = os.path.basename(file_path)
            name_no_ext = os.path.splitext(file_name)[0]
            
            # 1. Load
            df = load_data(file_path)
            
            # 2. Trim
            df = trim_data(df, limit=1800)
            
            # 3. Segment
            segments = segment_data(df, frames_per_segment=150)
            
            # Check if enough segments
            if len(segments) < 3:
                print(f"Skipping {file_name}: Not enough segments ({len(segments)} < 3)")
                error_count += 1
                continue
                
            # 5. Process Utterances (Seg 2+)
            utterances = segments[2:]
            processed_segments = []
            
            for seg in utterances:
                processed_seg = process_utterance_segment(seg)
                processed_segments.append(processed_seg)
                
            # 6. Reassemble
            # Notebook puts them back into a single DF? It calculates features per segment.
            # Goal is "Normalized Dataset". Likely means one normalized CSV per recording.
            # We will concatenate the processed utterance segments.
            # NOTE: We are intentionally dropping the reference segments from the output,
            # as they are just calibration. Or should we keep them? 
            # "remaining as the word utterance" implies the useful data is the utterances.
            # I will save the PROCESSED UTTERANCES only as they are the data of interest.
            
            final_df = pd.concat(processed_segments, ignore_index=True)
            
            # Create output dirs
            save_dir = os.path.join(output_root, word_dir)
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, file_name)
            final_df.to_csv(save_path, index=False)
            
            # 7. Plotting
            # Single plot of concatenated theta
            graph_save_dir = os.path.join(graph_root, word_dir)
            os.makedirs(graph_save_dir, exist_ok=True)
            graph_save_path = os.path.join(graph_save_dir, f"{name_no_ext}.png")
            
            plt.figure(figsize=(12, 4))
            plt.plot(final_df['theta'])
            plt.title(f"Normalized Theta: {word_dir} / {name_no_ext}")
            plt.xlabel("Frame")
            plt.ylabel("Theta (Norm)")
            plt.grid(True)
            plt.savefig(graph_save_path)
            plt.close()
            
            success_count += 1
            print(f"Processed: {rel_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_count += 1

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
