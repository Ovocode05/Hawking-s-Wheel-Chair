import numpy as np
import pandas as pd
import warnings
from scipy.signal import find_peaks as scipy_find_peaks

# --- Helper Functions (used only in this file) ---

def _find_peaks_simple(arr):
    """Fallback simple local-maximum peak finder (ignores flat peaks)."""
    if len(arr) < 3:
        return np.array([], dtype=int)
    left = arr[:-2]
    mid = arr[1:-1]
    right = arr[2:]
    peaks = np.where((mid > left) & (mid > right))[0] + 1
    return peaks

def _count_peaks(signal):
    """Count peaks using scipy.
    signal: 1d numpy array (NaNs removed)."""
    if len(signal) == 0:
        return 0
    try:
        prom = np.nanstd(signal) * 0.5
        prom = float(prom) if not np.isnan(prom) and prom > 0 else None
        if prom is not None:
            peaks, _ = scipy_find_peaks(signal, prominence=prom)
        else:
            peaks, _ = scipy_find_peaks(signal)
        return peaks.size
    except Exception:
        return _find_peaks_simple(signal).size

def _summary_stats(values):
    """Return simple summary: mean and standard deviation."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {"mean": np.nan, "std": np.nan}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}

# --- Core Processing Functions ---

def remove_initial_bad_and_compare(df, total_rows=1800, parts=12, initial_window=100, columns=None):
    """
    Cleans, segments, and processes a single DataFrame.
    Returns: avg_min, avg_max, diff, segments
    """
    if 'theta' not in df.columns:
        raise KeyError("DataFrame must contain a 'theta' column.")

    if len(df) < total_rows:
        warnings.warn(f"DataFrame has fewer than {total_rows} rows (got {len(df)}). Using all available rows.")
        total_rows = len(df)
    
    df_trim = df.iloc[:total_rows].reset_index(drop=True).copy()

    current_window = min(initial_window, total_rows)
    initial_idx = df_trim.index[:current_window]
    
    mask_bad = df_trim.loc[initial_idx, 'theta'].isna() | (df_trim.loc[initial_idx, 'theta'] == 0)
    bad_indices = df_trim.loc[initial_idx[mask_bad], :].index if mask_bad.any() else df_trim.index[[]]
    
    if len(bad_indices) > 0:
        df_trim = df_trim.drop(bad_indices).reset_index(drop=True)

    numeric_cols = df_trim.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df_trim[numeric_cols] = df_trim[numeric_cols].abs()

    if len(df_trim) < parts:
        raise ValueError(f"Not enough rows after removal to split into {parts} parts (have {len(df_trim)} rows).")

    total_kept = len(df_trim)
    base_size = total_kept // parts
    remainder = total_kept % parts

    segments = {}
    start = 0
    for i in range(parts):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        seg_df = df_trim.iloc[start:end].copy()
        name = f"reference_{i+1}" if i < 2 else f"measurement_{i-1}"
        seg_df['segment'] = i + 1
        seg_df['category'] = name
        segments[name] = seg_df
        start = end

    ref1 = segments.get("reference_1")
    ref2 = segments.get("reference_2")
    if ref1 is None or ref2 is None:
        raise KeyError("Expected both reference_1 and reference_2 segments present after splitting.")

    if columns is None:
        cols1 = ref1.select_dtypes(include=[np.number]).columns
        cols2 = ref2.select_dtypes(include=[np.number]).columns
        columns = list(cols1.intersection(cols2))
    
    columns = [col for col in columns if col in ref1.columns and col in ref2.columns]
    avg_min_series = ref1[columns].fillna(0).mean()
    avg_max_series = ref2[columns].fillna(0).mean()
    avg_min = avg_min_series.to_dict()
    avg_max = avg_max_series.to_dict()
    diff = {col: avg_max.get(col, 0) - avg_min.get(col, 0) for col in columns}

    return avg_min, avg_max, diff, segments

def motion_features_from_measurements(segments, measurement_prefix="measurement_", n_measurements=10, columns=("theta","x","y")):
    """
    Compute summary motion features for the measurement segments.
    """
    vals = {
        "num_peaks": [], "range_theta": [],
        "x_range": [], "x_disp": [],
        "y_range": [], "y_disp": []
    }
    
    if columns is None: columns = []

    for i in range(1, n_measurements+1):
        name = f"{measurement_prefix}{i}"
        seg = segments.get(name)
        if seg is None or len(seg) == 0:
            for k in vals.keys(): vals[k].append(np.nan)
            continue

        if "theta" in seg.columns and "theta" in columns:
            theta = np.abs(seg["theta"].dropna().to_numpy(dtype=float))
            vals["range_theta"].append((np.nanmax(theta) - np.nanmin(theta)) if theta.size > 0 else np.nan)
            vals["num_peaks"].append(_count_peaks(theta) if theta.size > 0 else np.nan)
        else:
            vals["range_theta"].append(np.nan); vals["num_peaks"].append(np.nan)

        if "x" in seg.columns and "x" in columns:
            x = np.abs(seg["x"].dropna().to_numpy(dtype=float))
            vals["x_range"].append((np.nanmax(x) - np.nanmin(x)) if x.size > 0 else np.nan)
            vals["x_disp"].append((x[-1] - x[0]) if x.size > 0 else np.nan)
        else:
            vals["x_range"].append(np.nan); vals["x_disp"].append(np.nan)

        if "y" in seg.columns and "y" in columns:
            y = np.abs(seg["y"].dropna().to_numpy(dtype=float))
            vals["y_range"].append((np.nanmax(y) - np.nanmin(y)) if y.size > 0 else np.nan)
            vals["y_disp"].append((y[-1] - y[0]) if y.size > 0 else np.nan)
        else:
            vals["y_range"].append(np.nan); vals["y_disp"].append(np.nan)
            
    final_summary = {}
    if "theta" not in columns:
        vals.pop("num_peaks", None); vals.pop("range_theta", None)
    if "x" not in columns:
        vals.pop("x_range", None); vals.pop("x_disp", None)
    if "y" not in columns:
        vals.pop("y_range", None); vals.pop("y_disp", None)

    for k, arr in vals.items():
        final_summary[k] = _summary_stats(np.asarray(arr, dtype=float))
    return final_summary