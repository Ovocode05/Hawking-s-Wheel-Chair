# Silent Speech Recognition - Jaw Kinematics Analysis

A project for silent speech recognition using jaw kinematics and motion data, focusing on medical and health-related vocabulary in Punjabi/Hindi.

## 📋 Overview

This project implements a silent speech recognition system that analyzes jaw movements to recognize spoken words without audible speech. The system processes kinematic data (theta, x, y, omega, alpha) from jaw movements and uses signal processing techniques to segment and classify words.

## 🎯 Project Purpose

The project aims to develop assistive communication technology for individuals who cannot speak audibly, similar to Stephen Hawking's communication system. It focuses on recognizing 25 common medical and health-related terms in Punjabi/Hindi.

## 📚 Vocabulary

The system recognizes the following 25 words:

**Health Conditions & Symptoms:**
- Bahaar (Outside)
- Bhukh/Bhookh (Hunger)
- Bukhar (Fever)
- Chah (Tea)
- Chakkar (Dizziness)
- Dard (Pain)
- Dhadkan/Dadkan (Heartbeat)
- Ghabrahat (Anxiety)
- Gharde (Family members)
- Jukham (Cold)
- Kabz (Constipation)
- Kamjori/Kamjoori (Weakness)
- Khangh (Cough)
- Khoon (Blood)
- Neend (Sleep)
- Piyaas (Thirst)
- Saah (Breath)
- Sardard (Headache)
- Takleef/Peed (Trouble/Pain)
- Ulti (Vomit)
- Dawai (Medicine)
- Doctor (Doctor)
- Hospital (Hospital)
- Paani/Panni (Water)
- Peshab (Urine)



## 🔬 Technical Approach

### 1. Data Collection
- Kinematic data captured from jaw movements
- Parameters tracked: theta (angle), x, y (position), omega (angular velocity), alpha (angular acceleration)
- Time-series CSV data with ~30 Hz sampling rate
- 5-second utterances per word

### 2. Signal Processing Pipeline

**Preprocessing:**
- Outlier removal using median ± 3*std clipping
- Interpolation for missing values
- Median filtering (kernel=5) for denoising
- Savitzky-Golay filtering for smooth peak detection

**Segmentation:**
- Peak detection in theta (jaw angle) signal
- Adaptive thresholding based on signal statistics
- Energy-based filtering to remove false peaks
- Minimum distance enforcement between peaks
- Window extraction (25 frames before, 50 frames after peak)

**Features:**
- Multi-dimensional kinematic features (theta, x, y, omega, alpha)
- Time-series segments aligned to detected peaks
- Statistical features (mean, std, max, min)

### 3. Classification
- Dynamic Time Warping (DTW) for template matching
- Euclidean distance in feature space
- Segment-to-segment comparison


## 📝 File Descriptions

- **seg(main).py**: Time-aware segmentation with expected utterance intervals
- **segmentation.py**: Complete robust pipeline with adaptive filtering
- **compare.py**: DTW-based word similarity comparison using basic DTW distance metric
- **compare2.py**: Multi-metric comparison system with 9 different distance measures

### seg(main).py - Time-Based Peak Selection Pipeline

This script implements a time-aware segmentation approach specifically designed for controlled recording sessions where words are spoken at regular intervals (approximately every 5 seconds).

**Key Features:**

**1. Data Preprocessing:**
- Column name cleaning (strip whitespace)
- Numeric conversion with error handling
- Forward interpolation for missing values
- Median filling for remaining NaN values
- Outlier clipping using median ± 3*std for all kinematic features

**2. Noise Reduction:**
- Median filtering (kernel=5) on all features (theta, x, y, omega, alpha)
- Savitzky-Golay smoothing (window=31, polynomial order=3) for peak detection
- Detrending using rolling median (window=301) to remove slow drift

**3. Expected Time-Based Peak Selection:**
```python
expected_times = np.arange(15, time[-1], 5)  # Peaks every 5 seconds starting at 15s
tolerance = 3.0                               # ±3 second search window
```
- Searches for peaks within ±3 seconds of expected utterance times
- Uses local peak detection with prominence threshold (0.08 * std)
- Selects highest peak within each time window
- Robust to timing variations in manual recordings

**4. Segment Extraction:**
- Window: 25 frames before peak, 50 frames after peak
- Z-score normalization per segment (mean=0, std=1)
- Handles all 5 kinematic features simultaneously
- Filters out segments shorter than 10 frames

**5. Visualization & EDA:**
- Signal plot with detected peaks and segment windows
- Theta distribution histogram with KDE
- Feature correlation heatmap
- Normalized segment overlays for theta and omega
- Color-coded segment windows (orange shading)

**6. Output:**
- Saves to `.npz` format with normalized segment arrays
- Compatible with downstream DTW comparison tools
- Object dtype array to handle variable-length segments

### compare2.py - Advanced Multi-Metric Analysis

This script provides comprehensive word comparison using 9 different signal analysis metrics, offering a more robust similarity assessment than DTW alone:

**Metrics Implemented:**

1. **DTW Distance** - Dynamic Time Warping on full 5-dimensional feature space (theta, x, y, omega, alpha)
2. **FFT Power Spectrum Distance** - Compares frequency domain representations using Welch's method
3. **Dominant Frequency Distance** - Measures difference in primary frequency components
4. **Spectral Entropy Distance** - Compares information content of frequency distributions
5. **Wavelet Distance** - Continuous wavelet transform comparison using Ricker wavelets (scales 1-30)
6. **Trend Distance** - Compares signal slopes using linear regression
7. **Seasonality Distance** - STL decomposition to compare periodic patterns (period=20 frames)
8. **Stationarity Distance** - Augmented Dickey-Fuller test statistic comparison
9. **Volatility Distance** - Short-term variance comparison (rolling window=10 frames)

**Features:**
- Segment-to-segment comparison across all metrics
- Word-level averaged results for comprehensive similarity assessment
- Visualization overlays of representative segments
- Robust error handling for edge cases (short segments, etc.)

**Output:**
- Detailed metric scores for each segment pair
- Average metric values per word comparison
- Overlay plots showing theta signal alignment

This multi-metric approach provides deeper insights into word similarity patterns beyond temporal alignment, capturing frequency characteristics, statistical properties, and signal dynamics.
