# ==================================================
# 🧠 Silent Speech (Jaw Kinematics) Data Pipeline
# Segmentation + Robust Peak Filtering + EDA + Save
# ==================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
import os

# ------------------------
# Parameters (tune if needed)
# ------------------------
data_path = r"C:\\Users\\lenovo\\Desktop\\Research2\\Hawking-s-Wheel-Chair\\Data10\\bahaar\\suresh.csv"
use_savgol = True               # smoother than rolling mean
savgol_win = 31                 # must be odd and <= len(signal)
savgol_poly = 3

prominence_factor = 0.2         # initial prominence = factor * std(signal)
initial_distance = 40           # minimum samples between peaks (initial)
merge_min_distance = 35         # merge very close clean peaks
final_min_distance = 60         # final grouping distance (frames)

height_rel_factor = 0.6         # initial relative height filter (0.6 * max peak height)

window_half_energy = 25         # frames on each side for energy calc
energy_ratio_thresh = 0.6       # keep peaks with energy > ratio * max_energy

# Thresholding: you requested hard cutoff at theta = 44 earlier.
# Keep min_theta=None to disable absolute cutoff, or set to 44 to use it.
min_theta = 44.0                # set to None to disable; otherwise peaks below this are ignored

# Adaptive threshold fraction (used in addition to min_theta)
adaptive_frac = 0.8             # keep peaks above adaptive_frac * max_peak_height

# Segment extraction windows
window_before, window_after = 25, 50
min_segment_length = 20         # ignore too-short segments

# Save path
save_dir = r"C:\\Users\\lenovo\\Desktop\\Research2\\Hawking-s-Wheel-Chair\\Data10\\bhookh"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "bhookh_segments_kinematics.npz")

# ==================================================
# 1. LOAD RAW DATA
# ==================================================
data = pd.read_csv(data_path)
print("✅ Data loaded successfully!")
print("Shape:", data.shape)
print("Columns:", list(data.columns))
print(data.head())

# ==================================================
# 2. CLEAN + SMOOTH
# ==================================================
data = data.dropna().reset_index(drop=True)

if use_savgol and len(data) >= savgol_win:
    data["theta_smooth"] = savgol_filter(data["theta"].values, window_length=savgol_win, polyorder=savgol_poly)
else:
    # fallback rolling mean
    data["theta_smooth"] = data["theta"].rolling(window=5, center=True).mean()
    data["theta_smooth"].fillna(data["theta"], inplace=True)

signal = data["theta_smooth"].values
time = data["t"].values

# ==================================================
# 3. INITIAL PEAK DETECTION
# ==================================================
prominence = prominence_factor * np.std(signal)
peaks, props = find_peaks(signal, prominence=prominence, distance=initial_distance)
print(f"🔎 Initial peaks found: {len(peaks)}")

# remove first calibration/silence peak if present
if len(peaks) > 0:
    peaks = peaks[1:]

if len(peaks) == 0:
    print("⚠️ No peaks found after initial removal. Exiting pipeline early.")
    filtered_peaks = np.array([], dtype=int)
else:
    # --- Filter by relative height (using detected peak heights) ---
    peak_heights = signal[peaks]
    max_height = np.max(peak_heights) if len(peak_heights) > 0 else np.max(signal)
    height_threshold = height_rel_factor * max_height
    mask = peak_heights > height_threshold
    clean_peaks = peaks[mask]

    if len(clean_peaks) == 0:
        print("⚠️ No peaks survive the relative-height filter. Falling back to raw peaks.")
        clean_peaks = peaks.copy()

    # --- Enforce min distance (merge very close ones) ---
    filtered_peaks = []
    for p in clean_peaks:
        if not filtered_peaks:
            filtered_peaks.append(p)
        else:
            if p - filtered_peaks[-1] > merge_min_distance:
                filtered_peaks.append(p)
            else:
                # keep the taller of the two
                if signal[p] > signal[filtered_peaks[-1]]:
                    filtered_peaks[-1] = p
    filtered_peaks = np.array(filtered_peaks, dtype=int)

print(f"🧩 Peaks after initial filtering & merging: {len(filtered_peaks)}")

# ==================================================
# 4. FINAL REFINEMENT: ENERGY-BASED + ADAPTIVE HEIGHT
# ==================================================
if len(filtered_peaks) == 0:
    final_peaks = np.array([], dtype=int)
else:
    # compute local energy for each filtered peak
    peak_energies = []
    for p in filtered_peaks:
        start = max(0, p - window_half_energy)
        end = min(len(signal), p + window_half_energy)
        baseline = np.median(signal[start:end]) if end - start > 0 else 0.0
        energy = np.sum((signal[start:end] - baseline) ** 2)
        peak_energies.append(energy)
    peak_energies = np.array(peak_energies)
    max_energy = peak_energies.max() if len(peak_energies) > 0 else 0.0

    # determine adaptive threshold based on peak heights
    peak_heights = signal[filtered_peaks]
    adaptive_threshold = adaptive_frac * np.max(peak_heights) if len(peak_heights) > 0 else -np.inf

    # combine with optional absolute min_theta
    if min_theta is not None:
        overall_height_thresh = max(min_theta, adaptive_threshold)
    else:
        overall_height_thresh = adaptive_threshold

    final_peaks = []
    for i, p in enumerate(filtered_peaks):
        height_ok = signal[p] >= overall_height_thresh
        energy_ok = (peak_energies[i] >= energy_ratio_thresh * max_energy) if max_energy > 0 else True

        # keep candidate only if it passes both height and energy tests
        if not (height_ok and energy_ok):
            continue

        # ensure distance from previous final peak; if close, keep the one with higher energy
        if final_peaks and (p - final_peaks[-1]) < final_min_distance:
            prev = final_peaks[-1]
            prev_idx = np.where(filtered_peaks == prev)[0][0]
            # choose which to keep based on energy
            if peak_energies[i] > peak_energies[prev_idx]:
                final_peaks[-1] = p
            # else keep previous, ignore current
        else:
            final_peaks.append(p)

    final_peaks = np.array(sorted(final_peaks), dtype=int)

print(f"✅ Final peaks retained after refinement: {len(final_peaks)}")
print(f"Adaptive height threshold used = {overall_height_thresh:.2f} (adaptive_frac={adaptive_frac}, min_theta={min_theta})")

# Plot raw peaks vs final peaks for debugging
plt.figure(figsize=(12, 5))
plt.plot(time, signal, label="θ (smoothed)", color="b")
if len(peaks) > 0:
    plt.scatter(time[peaks], signal[peaks], color="orange", label="Raw detected peaks", zorder=3)
if len(filtered_peaks) > 0:
    plt.scatter(time[filtered_peaks], signal[filtered_peaks], color="purple", label="After merge filter", zorder=3)
if len(final_peaks) > 0:
    plt.scatter(time[final_peaks], signal[final_peaks], color="red", label="Final peaks (kept)", zorder=4)
plt.axhline(overall_height_thresh, color="green", linestyle="--", alpha=0.6, label=f"Height threshold = {overall_height_thresh:.2f}")
plt.title("Peak Detection — raw → merged → final")
plt.xlabel("Time (s)")
plt.ylabel("θ (angle)")
plt.legend()
plt.tight_layout()
plt.show()

# ==================================================
# 5. SEGMENT EXTRACTION AROUND FINAL PEAKS
# ==================================================
segments = []
labels = []

eps = 1e-8
for p in final_peaks:
    start = max(0, p - window_before)
    end = min(len(data), p + window_after)
    seg_df = data.loc[start:end, ["theta", "x", "y", "omega", "alpha"]].values
    if len(seg_df) >= min_segment_length:
        # safe normalization (avoid zero std)
        mu = np.mean(seg_df, axis=0)
        sigma = np.std(seg_df, axis=0)
        sigma[sigma < eps] = 1.0
        seg_norm = (seg_df - mu) / sigma
        segments.append(seg_norm)
        labels.append("bahaar")

segments = np.array(segments, dtype=object)
labels = np.array(labels)

print(f"✅ Extracted {len(segments)} valid normalized segments")

# ==================================================
# 6. SAVE SEGMENTS + META
# ==================================================
meta = {
    "word": "bahaar",
    "file": os.path.basename(data_path),
    "n_segments": len(segments),
    "final_peak_indices": final_peaks.tolist(),
    "params": {
        "min_theta": min_theta,
        "adaptive_frac": adaptive_frac,
        "energy_ratio_thresh": energy_ratio_thresh
    }
}

np.savez(save_path, X=segments, y=labels, meta=meta)
print(f"💾 Segments and meta saved to: {save_path}")

# ==================================================
# 7. EDA ON SEGMENTED DATA (same as before, robustified)
# ==================================================
feature_cols = ["theta", "x", "y", "omega", "alpha"]

if len(segments) == 0:
    print("⚠️ No segments to analyze. Pipeline finished.")
else:
    segment_lengths = [len(s) for s in segments]
    plt.figure(figsize=(8, 4))
    sns.histplot(segment_lengths, bins=15, kde=True)
    plt.title("Distribution of Segment Lengths")
    plt.xlabel("Frames per Segment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    print("📏 Segment Lengths Summary:")
    print(pd.Series(segment_lengths).describe())

    # overlay θ and ω
    plt.figure(figsize=(12, 5))
    for seg in segments:
        plt.plot(seg[:, 0], alpha=0.4)
    plt.title("Overlay of All θ Segments (normalized)")
    plt.xlabel("Frame Index")
    plt.ylabel("θ (normalized)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for seg in segments:
        plt.plot(seg[:, 3], alpha=0.4)
    plt.title("Overlay of All ω Segments (normalized)")
    plt.xlabel("Frame Index")
    plt.ylabel("ω (normalized)")
    plt.tight_layout()
    plt.show()

    # Detailed EDA for segment 0 (if exists)
    seg_idx = 0
    seg_df = pd.DataFrame(segments[seg_idx], columns=feature_cols)
    print(f"\n📈 EDA for segment #{seg_idx} | Shape: {seg_df.shape}")

    plt.figure(figsize=(12, 8))
    for i, col in enumerate(feature_cols):
        plt.subplot(len(feature_cols), 1, i + 1)
        plt.plot(seg_df[col], label=col)
        plt.ylabel(col)
        plt.legend(loc='upper right')
    plt.suptitle(f"Feature Trends for Segment #{seg_idx}", y=1.02)
    plt.xlabel("Frame Index")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(seg_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Feature Correlation Heatmap (Segment #{seg_idx})")
    plt.tight_layout()
    plt.show()

    seg_melt = seg_df.melt(var_name="Feature", value_name="Value")
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=seg_melt, x="Feature", y="Value", inner="quartile", palette="viridis")
    plt.title(f"Feature Distributions per Variable (Segment #{seg_idx})")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(seg_df["theta"], seg_df["omega"], c=np.arange(len(seg_df)), cmap="plasma")
    plt.colorbar(label="Frame Index")
    plt.title(f"Phase Plot: θ vs ω (Segment #{seg_idx})")
    plt.xlabel("θ (angle)")
    plt.ylabel("ω (angular velocity)")
    plt.tight_layout()
    plt.show()
    
    # summary stats
    summary = seg_df.describe().T
    summary["variance"] = seg_df.var()
    summary["range"] = seg_df.max() - seg_df.min()
    cols = ["mean", "std", "min", "max", "variance", "range"]
    print("\n📊 Feature Summary for Segment:")
    print(summary[cols])
