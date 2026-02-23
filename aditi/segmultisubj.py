import pandas as pd
import numpy as np
from scipy.signal import medfilt, savgol_filter, find_peaks
import matplotlib.pyplot as plt
from pathlib import Path

# ================= PATHS =================
root_dir = Path(r'D:\Desktop\Research2\Data3')
output_dir = Path(r'D:\Desktop\Research2\Krrish3npz')
vis_dir = output_dir / "visualizations"

output_dir.mkdir(exist_ok=True)
vis_dir.mkdir(exist_ok=True)

# ================= VISUALIZATION =================
def plot_jaw_angle_with_segments(word, time, signal, peaks, save_dir,
                                 window_before=50, window_after=50):

    plt.figure(figsize=(14, 6))
    plt.plot(time, signal, label='Smoothed θ')

    for i, p in enumerate(peaks):
        plt.axvline(x=time[p], color='red', linestyle='--', alpha=0.6)
        plt.scatter(time[p], signal[p], color='red', zorder=5,
                    label='Peak' if i == 0 else "")

        start = max(0, p - window_before)
        end = min(len(signal) - 1, p + window_after)

        plt.axvspan(time[start], time[end],
                    color='orange', alpha=0.25,
                    label='Segment window' if i == 0 else "")

    plt.title(f'Jaw Angle with Selected Peaks and Segment Windows – {word}')
    plt.xlabel('Time')
    plt.ylabel('θ (angle)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{word}_jaw_peaks_segments.png")
    plt.close()

# ================= MAIN LOOP =================
for word_dir in root_dir.iterdir():

    if not word_dir.is_dir():
        continue

    word_label = word_dir.name
    print(f"\n🟢 Processing word: {word_label}")

    word_segments = []

    rep_time = None
    rep_signal = None
    rep_peaks = None

    
    csv_file = word_dir / "Krrish.csv"
    if not csv_file.exists():
        print("csv not found, skipping word")
        continue

    print(f"   └─ {csv_file.name}")

    # ---------- LOAD ----------
    data = pd.read_csv(csv_file, na_values=['', ' '], skipinitialspace=True)
    data.columns = data.columns.str.strip().str.lower()

    if 'theta' not in data.columns:
        print("     ⚠ theta column missing, skipping")
        continue

    # ---------- NUMERIC ----------
    data = data.apply(pd.to_numeric, errors='coerce')

    # allow delayed theta start
    valid_theta_ratio = data['theta'].notna().mean()
    if valid_theta_ratio < 0.05:
        print("     ⚠ theta too sparse, skipping")
        continue

    # ---------- FILL ----------
    data = data.interpolate(limit_direction='forward')
    data.fillna(data.median(), inplace=True)

    # ---------- FILTER ----------
    for col in ['theta', 'x', 'y', 'omega', 'alpha']:
        if col in data.columns:
            med, std = data[col].median(), data[col].std()
            data[col] = data[col].clip(med - 3*std, med + 3*std)
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

    time = data['t'].values if 't' in data.columns else np.arange(len(data))

    # ---------- PEAK DETECTION ----------
    expected_times = np.arange(15, time[-1], 5)
    tolerance = 3
    min_prom = 0.08 * np.std(signal)

    selected_peaks = []
    for et in expected_times:
        mask = (time >= et - tolerance) & (time <= et + tolerance)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue

        local_sig = signal[idxs]
        peaks, _ = find_peaks(
            local_sig,
            prominence=min_prom,
            distance=10
        )

        if len(peaks):
            best = peaks[np.argmax(local_sig[peaks])]
            selected_peaks.append(idxs[best])

    selected_peaks = sorted(set(selected_peaks))
    if len(selected_peaks) > 1:
        selected_peaks = selected_peaks[1:]  # remove first peak
        #selected_peaks = selected_peaks[:-1]
    # ---------- SEGMENTS ----------
    for p in selected_peaks:
        start, end = max(0, p - 50), min(len(data) - 1, p + 50)
        seg = data.loc[start:end,
                       ['theta', 'x', 'y', 'omega', 'alpha']].values

        if len(seg) > 10:
            mu, sigma = seg.mean(axis=0), seg.std(axis=0)
            sigma[sigma < 1e-8] = 1.0
            word_segments.append((seg - mu) / sigma)

    # ---------- VISUALIZATION ----------
    if len(selected_peaks) > 0:
        plot_jaw_angle_with_segments(
            word=word_label,
            time=time,
            signal=signal,
            peaks=selected_peaks,
            save_dir=vis_dir
        )

    # ---------- SAVE ----------
    word_segments = np.array(word_segments, dtype=object)

    np.savez(
        output_dir / f"{word_label}.npz",
        segments=word_segments,
        label=word_label
    )

    print(f"✅ Saved {len(word_segments)} segments for '{word_label}'")

print("\n🎉 All words processed successfully!")
