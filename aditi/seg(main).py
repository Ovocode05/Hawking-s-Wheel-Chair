import pandas as pd
import numpy as np
from scipy.signal import medfilt, savgol_filter, find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load and prepare data ---
data_path = r'C:\\Users\\lenovo\\Desktop\\Research2\\Hawking-s-Wheel-Chair\\Data10\\Gharde\\suresh.csv'
data = pd.read_csv(data_path, delimiter=',')  # Use delimiter=',' for your format

data.columns = data.columns.str.strip()  # Remove leading/trailing spaces

# Ensure 'theta' is present
if 'theta' not in data.columns:
    raise Exception("'theta' column not found! Columns are:", data.columns.tolist())

# Fill missing with interpolation and then median
data = data.apply(pd.to_numeric, errors='coerce')
data = data.interpolate(limit_direction='forward')
data.fillna(data.median(), inplace=True)

# Outlier clipping for numerical columns
for col in ['theta', 'x', 'y', 'omega', 'alpha']:
    if col in data.columns:
        med, std = data[col].median(), data[col].std()
        data[col] = data[col].clip(med - 3*std, med + 3*std)

# Denoise with median filter (kernel=5)
for col in ['theta', 'x', 'y', 'omega', 'alpha']:
    if col in data.columns:
        data[col] = medfilt(data[col], kernel_size=5)

# Smoothing for peak detection
if len(data) >= 31:
    data['theta_smooth'] = savgol_filter(data['theta'], window_length=31, polyorder=3)
else:
    data['theta_smooth'] = data['theta'].rolling(window=5, center=True).mean()
    data['theta_smooth'].fillna(data['theta'], inplace=True)

# --- Light detrend to remove slow drift / offsets ---
signal = data['theta_smooth'].values
signal = signal - pd.Series(signal).rolling(301, center=True, min_periods=1).median().values

# Use 't' (time) if present, otherwise just row indices
time = data['t'].values if 't' in data.columns else np.arange(len(data))

# --- Select peaks near expected times using local search ---
expected_times = np.arange(15, time[-1], 5)
tolerance = 3.0            # widen window
min_prom = 0.08 * np.std(signal)  # smaller prominence

selected_peaks = []
for et in expected_times:
    mask = (time >= et - tolerance) & (time <= et + tolerance)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        continue
    local_sig = signal[idxs]
    peaks, _ = find_peaks(local_sig, prominence=min_prom, distance=10)
    if len(peaks):
        best_local = peaks[np.argmax(local_sig[peaks])]
        selected_peaks.append(idxs[best_local])

selected_peaks = np.array(sorted(set(selected_peaks)))
print("Final selected peaks indices:", selected_peaks)
print("Corresponding times:", time[selected_peaks])
# --- Visualization of peaks/windows ---
window_before = 25
window_after = 50

plt.figure(figsize=(14, 6))
plt.plot(time, signal, label='Smoothed θ')
for i, p in enumerate(selected_peaks):
    plt.axvline(x=time[p], color='red', linestyle='--', alpha=0.6)
    plt.scatter(time[p], signal[p], color='red', zorder=5, label='Peak' if i==0 else "")
    start_idx = max(0, p - window_before)
    end_idx = min(len(signal)-1, p + window_after)
    plt.axvspan(time[start_idx], time[end_idx], color='orange', alpha=0.2, label='Segment window' if i==0 else "")
plt.title('Jaw Angle with Selected Peaks and Segment Windows')
plt.xlabel('Time')
plt.ylabel('θ (angle)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Segment Extraction and Normalization ---
segments = []
for p in selected_peaks:
    start, end = max(0, p - window_before), min(len(data)-1, p + window_after)
    seg = data.loc[start:end, ['theta', 'x', 'y', 'omega', 'alpha']].values
    if len(seg) > 10:  # only keep reasonably sized segments
        mu, sigma = np.mean(seg, axis=0), np.std(seg, axis=0)
        sigma[sigma < 1e-8] = 1.0  # to avoid zero division
        seg_norm = (seg - mu) / sigma
        segments.append(seg_norm)
segments = np.array(segments, dtype=object)
print(f"Extracted {len(segments)} normalized segments.")

# --- EDA ---
plt.figure(figsize=(10, 4))
sns.histplot(data['theta_smooth'], bins=40, kde=True, color='steelblue')
plt.title('Distribution of Smoothed Theta')
plt.xlabel('θ')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(data[['theta', 'x', 'y', 'omega', 'alpha']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

plt.figure(figsize=(12, 5))
for seg in segments:
    plt.plot(seg[:, 0], alpha=0.3)  # theta
plt.title('Overlay of Theta Segments (Normalized)')
plt.xlabel('Frame Index')
plt.ylabel('Normalized θ')
plt.show()

plt.figure(figsize=(12, 5))
for seg in segments:
    plt.plot(seg[:, 3], alpha=0.3)  # omega
plt.title('Overlay of Omega Segments (Normalized)')
plt.xlabel('Frame Index')
plt.ylabel('Normalized ω')
plt.show()

# --- Save segments for model training ---
np.savez('d10Gharde.npz', segments=segments)
print("Saved normalized segments to 'd10Gharde.npz'")