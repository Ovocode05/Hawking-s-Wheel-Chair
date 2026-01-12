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

def process_utterance_segment(segment):
    """
    Applies KNN Imputation and Z-score Normalization to a single segment.
    """
    # 1. Imputation
    if segment.isnull().values.any():
        imputer = KNNImputer(n_neighbors=5)
        # KNNImputer returns numpy array, reconstruct DataFrame
        imputed_data = imputer.fit_transform(segment)
        segment_clean = pd.DataFrame(imputed_data, columns=segment.columns, index=segment.index)
    else:
        segment_clean = segment.copy()

    # 2. Normalization (Z-score)
    # theta_norm = (theta - mean) / std
    if segment_clean['theta'].std() != 0:
        segment_clean['theta'] = (segment_clean['theta'] - segment_clean['theta'].mean()) / segment_clean['theta'].std()
    else:
        # Handle case where std is 0 (all values are the same)
        segment_clean['theta'] = 0.0
    
    return segment_clean
