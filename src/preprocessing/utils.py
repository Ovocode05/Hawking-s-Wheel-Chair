import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def load_data(filepath):
    """Loads CSV data from the given filepath."""
    return pd.read_csv(filepath)

def trim_data(df, limit=1800):
    """Trims the dataframe to the specified number of rows."""
    return df.head(limit)

def segment_data(df, frames_per_segment=150):
    """Splits the dataframe into list of segments."""
    total_frames = len(df)
    n_segments = total_frames // frames_per_segment
    segments = []
    for i in range(n_segments):
        start = i * frames_per_segment
        end = start + frames_per_segment
        segments.append(df.iloc[start:end].copy().reset_index(drop=True))
    return segments

def calculate_references(ref_min_seg, ref_max_seg):
    """
    Calculates reference minimum and maximum angles based on the first two segments.
    Logic matches notebook:
    - Min: Mean of theta > 0 in ref_min_seg (after filling NaNs with 0 temporarily)
    - Max: Max of theta in ref_max_seg
    """
    # Min calculation
    ref_min_filled = ref_min_seg.fillna(0.0)
    # Filter for theta > 0
    ref_angle_min = ref_min_filled[ref_min_filled["theta"] > 0]["theta"].mean()
    
    # Max calculation
    ref_angle_max = ref_max_seg["theta"].max()
    
    return ref_angle_min, ref_angle_max

def process_utterance_segment(segment, ref_min, ref_max):
    """
    Applies KNN Imputation and Min-Max Normalization to a single segment.
    """
    # 1. Imputation
    if segment.isnull().values.any():
        imputer = KNNImputer(n_neighbors=5)
        # KNNImputer returns numpy array, reconstruct DataFrame
        # We only strictly need to impute 'theta' but KNN uses other columns? 
        # Notebook applies fit_transform to segment_df.copy() which implies all numeric columns.
        # We assume the segment has numeric columns compatible with KNN (frame_idx, t, theta, x, y, omega, alpha)
        # If text columns exist they would error, but data seems numeric.
        imputed_data = imputer.fit_transform(segment)
        segment_clean = pd.DataFrame(imputed_data, columns=segment.columns, index=segment.index)
    else:
        segment_clean = segment.copy()

    # 2. Normalization
    # theta_norm = (theta - ref_min) / (ref_max - ref_min)
    # Note: Notebook plots "Normalized Theta (0-1 Scale)"
    segment_clean['theta'] = (segment_clean['theta'] - ref_min) / (ref_max - ref_min)
    
    return segment_clean
