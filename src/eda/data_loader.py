import os
import pandas as pd
# Notice the relative import:
from .feature_extraction import remove_initial_bad_and_compare, motion_features_from_measurements

def gather_motion_summaries(words, names, base_path="Data/recordings", file_ext=".csv",
                            total_rows=1800, parts=12, initial_window=100, columns=("theta","x","y")):
    """
    Read each Data/recordings/{word}/{name}.csv, run cleaning + segmentation, extract
    motion summary (mean/std per metric) using existing functions.
    """
    summaries = {}
    errors = []
    
    if columns is None: columns = ()
    elif isinstance(columns, str): columns = (columns,)
    else: columns = tuple(columns)

    for w in words:
        summaries[w] = {}
        for n in names:
            path = os.path.join(base_path, w, f"{n}{file_ext}")
            if not os.path.isfile(path):
                errors.append((w, n, "file missing"))
                continue
            try:
                df_local = pd.read_csv(path)
                # Call the functions from feature_extraction.py
                _, _, _, segments = remove_initial_bad_and_compare(df_local, total_rows=total_rows,
                                                                  parts=parts, initial_window=initial_window,
                                                                  columns=columns)
                
                summary = motion_features_from_measurements(segments, measurement_prefix="measurement_",
                                                          n_measurements=parts-2, columns=columns)
                
                summaries[w][n] = summary
            except Exception as e:
                errors.append((w, n, str(e)))
    return summaries, errors