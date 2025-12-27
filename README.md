# Jaw-Motion Pipeline — Concise Technical Summary

This document summarizes the complete pipeline, analyses, outputs, and next steps for the jaw-motion recordings project.

---

## 1. EDA directory structure

**Core code**
- `feature_extraction.py` — cleaning, segmentation, feature extraction (per-segment → per-recording mean/std)
- `data_loader.py` — aggregate per-word / per-subject summaries from CSVs
- `analysis.py` — intersubject variability, Fisher score, Mahalanobis distance
- `visualization.py` — heatmaps, barplots, similarity matrices
- `main.py` — automated pipeline runner and plotting harness

**Data / outputs**
- `Data/recordings/{word}/{subject}.csv` — raw recordings
- `Graphs/` — plots (raw vs cleaned, correlations, heatmaps, etc.)
- `results/` — tabular outputs from analyses
- `Graphs/` also contains raw CV tracking plots from computer vision

---

## 2. Automated preprocessing pipeline

For each CSV (or DataFrame):

1. Read recording.
2. Trim to first `N=1800` rows (warn if fewer).
3. In first 100 rows: drop rows where `theta` is NaN or zero.
4. Take absolute value of numeric columns.
5. Split into 12 parts:
   - `reference_1` (part 1) → minimum reference  
   - `reference_2` (part 2) → maximum reference  
   - `measurement_1` … `measurement_10` (parts 3–12) → motion segments
6. Automated preprocessing:
   - Fill short NaN gaps only (sample-limited).
   - Smooth (Savitzky–Golay / median / rolling).
   - Detect outliers using robust MAD thresholds.
   - Replace outliers with smoothed values.
   - Return cleaned data + boolean masks (`filled`, `outlier`, `changed`) and per-column counts.

---

## 3. Extracted motion features (per segment → summarized)

For each of the 10 measurement segments:

- `num_peaks` (theta)
- `range_theta = max(theta) − min(theta)`
- `x_range`, `y_range`
- `x_disp = last(x) − first(x)`
- `y_disp = last(y) − first(y)`

Per recording / subject / word: **mean and standard deviation** of each metric across the 10 segments.

---

## 4. Analyses implemented

### 4.1 Intersubject variability (within a word)

For each word × metric:
- mean of subject means
- std of subject means
- coefficient of variation `cv = std / mean`

Output: DataFrame (words × metrics) and CV heatmap.  
Interpretation: higher CV → higher subject-to-subject variability.

---

### 4.2 Word separability

- **Per-metric Fisher score** = between-word variance / within-word variance.
- **Pairwise Mahalanobis distance** between word centroids (using pooled within-class covariance).

Outputs:
- `fisher_df` (metric importance)
- `pairwise_mahalanobis` (word × word distance matrix)
- `centroids` (word × metric mean vectors)

Interpretation: higher Fisher or Mahalanobis → better separability; small distances indicate confusable words.

---

### 4.3 Within-subject word similarity

For a selected subject:
- Build per-word feature vectors.
- Compute pairwise distances and similarity `1 / (1 + dist)`.

Outputs: distance and similarity matrices (heatmaps).  
Interpretation: lower distance / higher similarity → similar articulation for that subject.

---

## 5. Visual outputs

Saved in `Graphs/`:
- Raw vs cleaned time-series comparisons
- Cleaned correlation plots
- Intersubject CV heatmap
- Fisher score barplot
- Mahalanobis distance heatmap
- Within-subject similarity heatmap

Example filenames:
- `Graphs/Amish_refined_comparison.png`
- `Graphs/Amish_refined_correlation.png`
- `Graphs/word_separability_fisher_score.png`
- `Graphs/word_separability_distance_matrix.png`

---

## 6. Example usage

**Gather summaries**
```python
from eda_src.data_loader import gather_motion_summaries
summaries, errors = gather_motion_summaries(words, names, base_path="Data/recordings")
```

**Run analyses**
```python
from eda_src.analysis import compute_word_similarity_for_subject, intersubject_variability_table, word_separability_metrics
from eda_src.visualization import plot_intersubject_variability, plot_word_separability, plot_within_subject_similarity

results = produce_reports(words, names, base_path="Data/recordings", subject_for_word_similarity="Bansbir")
```

**Run preprocessing test**
```bash
python Preprocessing/main.py
```

---

## 7. Key findings (qualitative)

- Preprocessing robustly fills short gaps and suppresses outliers while tracking all modifications.
- Some metrics show high intersubject CV → likely sensitive to articulation or sensor placement.
- Fisher scores identify the most discriminative features (e.g., `range_theta` vs `x_disp`).
- Small Mahalanobis distances flag potentially confusable word pairs.

---

## 8. Limitations & next steps

- Use time-aware interpolation (Kalman / state-space) for longer gaps.
- Propagate imputation masks into downstream analysis and down-weight heavily imputed segments.
- Expand feature set (velocity, acceleration, frequency/time-frequency).
- Quantify separability via classifiers (LDA, logistic regression, random forest) using top Fisher features.
- Add PCA projections with centroid ± covariance ellipses.

---

## 9. Notes

- Expected data layout: `Data/recordings/{word}/{subject}.csv`
- Outputs default to `Graphs/` and `results/`.
- SciPy required for Savitzky–Golay and peak detection (fallbacks exist).
