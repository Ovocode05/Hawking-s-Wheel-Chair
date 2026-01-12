# Project Documentation: Jaw Motion EDA & Analysis

This document summarizes the exploratory data analysis, preprocessing pipeline, and distance metric analysis performed on the jaw-motion recordings.

## Dataset Selection
**Selected subset for analysis:**
*   **Words**: `Bahaar`, `Bhuk`, `Doctor`, `Saah`, `Neend`
*   **Subjects**: `Amish`, `anubhavjot`, `Jaskaran`, `Armman`

---

## Phase 1: Preprocessing & Feature Extraction
**Source**: `src/preprocessing/`, `src/eda/`

### Preprocessing Pipeline
1.  **Data Loading**: Ingests CSV recordings.
2.  **Cleaning & Imputation**:
    *   Trims to fixed length (e.g., 1800 frames).
    *   **KNN Imputer**: Used to fill missing values and handle gaps in the sensor data to ensure continuity.
    *   Removes artifacts (zero/NaN removal).
3.  **Normalization**: 
    *   Applies min-max or standard normalization to scale sensor values (theta, x, y) to a comparable range across subjects.
4.  **Segmentation**: Splits motion into reference and active measurement segments.

### Feature Extraction
Extracts a comprehensive set of features to characterize jaw motion:
*   **Time-Domain Features**:
    *   `num_peaks`: Frequency of peaks in theta (speed of fluctuations).
    *   `range_theta`: Full range of angular motion.
    *   `x_disp`, `y_disp`: Displacement ranges.
*   **Frequency-Domain Features**:
    *   Spectral features (FFT-based) to analyze the periodicity and frequency components of the jaw tremor/motion.

---

## Phase 2: Variability & Distance Analysis (DTW/JS)
**Source**: `src/eda-2/`

### Objective
Quantify inter-subject variability and inter-word distinction using advanced time-series and probabilistic metrics.

### Metrics
1.  **Dynamic Time Warping (DTW)**: Measures similarity between temporal sequences, accounting for speed variations in speech/motion.
2.  **Jensen-Shannon Divergence (JS)**: Measures similarity between probability distributions of feature values.

### Implementation Modules

#### 1. Distance Analysis (`src/eda-2/analysis_dtw_js.py`)
This module performs pairwise comparisons to evaluate:
*   **Inter-Subject Variability**: Calculates pairwise DTW/JS distances between different subjects saying the *same word*.
    *   **Visualization**: Boxplots (X-axis: Word, Y-axis: Distance). Lower spread/median indicates higher consistency.
*   **Inter-Word Distinction**: Calculates pairwise distances between *different words* spoken by the same subject.
    *   **Visualization**: Average Heatmaps (7x7 Matrix). Higher values indicate better distinguishability.
*   **Separability**: Kernel Density Plots comparing the distributions of "Same Word" variance vs "Different Word" distance.

#### 2. Hierarchical Clustering (`src/eda-2/dendogram.py`)
*   Uses JS Divergence matrices to perform hierarchical clustering.
*   **Visualization**: Dendrogram showing similarity groups (clades) of words based on their motion distribution features.

### Key Outputs (Graphs)
*   **Variability**: `dtw_inter_subject_boxplot.png`
*   **Distinction**: `dtw_inter_word_avg_heatmap.png`, `js_inter_word_avg_heatmap.png`
*   **Clustering**: Dendrogram plots (e.g., grouping `Doctor` with phonetically similar words).
*   **Separability**: `dtw_separability.png` (Comparison of Intra-class vs Inter-class distances).
