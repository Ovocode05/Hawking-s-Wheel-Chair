import pandas as pd
import numpy as np
from scipy.signal import medfilt, savgol_filter, find_peaks
import matplotlib.pyplot as plt
from pathlib import Path

# ================= PATHS =================
root_dir = Path(r'D:\Desktop\Research2\Data3')
output_dir = Path(r'D:\Desktop\Research2\segmented_data_trial\subject3npz')
vis_dir = output_dir / "visualizations"

output_dir.mkdir(exist_ok=True)
vis_dir.mkdir(exist_ok=True)

# ================= CONFIG =================
WINDOW_BEFORE = 50
WINDOW_AFTER = 50
EXPECTED_START = 15     # seconds
EXPECTED_STEP = 5       # seconds
EXPECTED_TOL = 3        # seconds

# ================= VISUALIZATION =================
def plot_jaw_angle_with_segments(word, time, signal, peaks, save_dir):

    plt.figure(figsize=(14, 6))
    plt.plot(time, signal, label='Smoothed θ')

    for i, p in enumerate(peaks):
        plt.axvline(x=time[p], color='red', linestyle='--', alpha=0.6)
        plt.scatter(time[p], signal[p], color='red', zorder=5,
                    label='Peak' if i == 0 else "")

        start = max(0, p - WINDOW_BEFORE)
        end = min(len(signal) - 1, p + WINDOW_AFTER)

        plt.axvspan(time[start], time[end],
                    color='orange', alpha=0.25,
                    label='Segment window' if i == 0 else "")

    plt.title(f'Jaw Angle with Segments – {word}')
    plt.xlabel('Time')
    plt.ylabel('θ (angle)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{word}_jaw_peaks_segments.png")
    plt.close()

# ================= MAIN LOOP (SUBJECT-WISE) =================
for word_dir in root_dir.iterdir():

    if not word_dir.is_dir():
        continue

    word_label = word_dir.name
    print(f"\n🟢 Processing: {word_label}")

    csv_file = word_dir / "Krrish.csv"
    if not csv_file.exists():
        print("csv not found, skipping")
        continue

    # ---------- LOAD ----------
    data = pd.read_csv(csv_file, na_values=['', ' '], skipinitialspace=True)
    data.columns = data.columns.str.strip().str.lower()

    if 'theta' not in data.columns:
        print("⚠ theta missing, skipping")
        continue

    data = data.apply(pd.to_numeric, errors='coerce')

    if data['theta'].notna().mean() < 0.05:
        print("⚠ theta too sparse, skipping")
        continue

    # ---------- FILL ----------
    data = data.interpolate(limit_direction='forward')
    data.fillna(data.median(), inplace=True)

    # ---------- FILTER ----------
    for col in ['theta', 'x', 'y', 'omega', 'alpha']:
        if col in data.columns:
            med, std = data[col].median(), data[col].std()
            data[col] = data[col].clip(med - 3 * std, med + 3 * std)
            data[col] = medfilt(data[col], kernel_size=5)

    # ---------- SMOOTH ----------
    if len(data) >= 31:
        data['theta_smooth'] = savgol_filter(data['theta'], 31, 3)
    else:
        data['theta_smooth'] = data['theta']

    # ---------- DETREND ----------
    signal = data['theta_smooth'].values
    signal -= pd.Series(signal).rolling(
        301, center=True, min_periods=1
    ).median().values

    time = data['t'].values if 't' in data.columns else np.arange(len(signal))

    # ---------- PERIODIC-AWARE PEAK SELECTION ----------
    expected_times = np.arange(EXPECTED_START, time[-1], EXPECTED_STEP)
    min_prom = 0.08 * np.std(signal)

    selected_peaks = []

    for et in expected_times:
        mask = (time >= et - EXPECTED_TOL) & (time <= et + EXPECTED_TOL)
        idxs = np.where(mask)[0]

        if len(idxs) == 0:
            continue

        local_signal = signal[idxs]
        peaks, props = find_peaks(
            local_signal,
            prominence=min_prom,
            distance=30
        )

        if len(peaks) == 0:
            continue

        best = peaks[np.argmax(props["prominences"])]
        p = idxs[best]

        if p - WINDOW_BEFORE < 0 or p + WINDOW_AFTER >= len(signal):
            continue

        selected_peaks.append(p)

    selected_peaks = sorted(set(selected_peaks))

    # ---------- BLIND FIRST PEAK REMOVAL ----------
    """if len(selected_peaks) > 1:
        selected_peaks = selected_peaks[1:]

    if len(selected_peaks) == 0:
        print("⚠ No valid peaks after removal")
        continue"""

    # ---------- SEGMENT EXTRACTION ----------
    word_segments = []
    for p in selected_peaks:
        start, end = p - WINDOW_BEFORE, p + WINDOW_AFTER
        seg = data.loc[start:end,
                       ['theta', 'x', 'y', 'omega', 'alpha']].values

        if len(seg) >= 30:
            word_segments.append(seg)

    word_segments = np.array(word_segments, dtype=object)

    # ---------- VISUALIZATION ----------
    plot_jaw_angle_with_segments(
        word=word_label,
        time=time,
        signal=signal,
        peaks=selected_peaks,
        save_dir=vis_dir
    )

    # ---------- SAVE ----------
    np.savez(
        output_dir / f"{word_label}.npz",
        segments=word_segments,
        label=word_label
    )

    print(f"✅ Saved {len(word_segments)} segments")

print("\n🎉 All words processed successfully!")
