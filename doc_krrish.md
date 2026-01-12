# Rough EDA / Pipeline summary

This is a concise summary of the exploratory data analysis and preprocessing work done so far on the jaw-motion recordings. It outlines the pipeline, the extracted features, analysis performed, key outputs, and suggested next steps.

## Project layout (relevant files)

- eda_src/feature_extraction.py — cleaning, segmentation, feature extraction (per measurement summary: mean/std).
- eda_src/data_loader.py — gather per-word / per-subject summaries from CSVs.
- eda_src/analysis.py — intersubject variability and word separability metrics (Fisher score, Mahalanobis).
- eda_src/visualization.py — plotting helpers (heatmaps, barplots, similarity matrices).
- Preprocessing/main.py — automated pipeline + plotting test harness.
- Data/recordings/... — CSV recordings organized by word/subject.
- Graphs/ — output plots saved here by scripts.

## Processing pipeline (automated)

1. Read CSV for a single recording (or accept DataFrame).
2. Trim to first N rows (default 1800); warn if fewer available.
3. Remove bad initial rows: within the first 100 rows drop rows where `theta` is NaN or zero.
4. Take absolute values of numeric columns (to make directionless metrics).
5. Split remaining rows into 12 parts:
   - reference_1 (part 1) → treated as "minimum" reference
   - reference_2 (part 2) → treated as "maximum" reference
   - measurement_1 .. measurement_10 (parts 3..12) → motion segments (jaw range of motion)

## Extracted motion features (per recording)

For the 10 measurement segments we extract per-segment metrics and then summarize across segments (subject/word-level):

- num_peaks (theta) — number of local maxima (peak count)
- range_theta — max(theta) − min(theta) in a segment
- x_range, y_range — spatial ranges for x and y
- x_disp, y_disp — displacement (last − first) in x and y

Aggregation currently uses mean and standard deviation across the 10 measurements for each recording/subject/word.

## Cross-subject / cross-word analysis

Two main analyses implemented:

1. Intersubject variability

   - For each word and each metric we compute:
     - mean of subject-means
     - std of subject-means
     - coefficient of variation (cv = std / mean)
   - Output: DataFrame (words × metrics) with mean / std / cv; visualized as a CV heatmap.
   - Interpretation: Higher CV → more variability across subjects for the same word.

2. Word separability (inter-word distinction)

   - Per-metric Fisher score: ratio of between-word variance to within-word variance (higher → better feature)
   - Multivariate pairwise Mahalanobis distance between word centroids (uses pooled within-class covariance)
   - Outputs:
     - fisher_df: Fisher score per metric
     - pairwise_mahalanobis: word × word distance matrix
     - centroids: word × metric mean vector
   - Interpretation: Higher Fisher and larger Mahalanobis distance indicate features/words that are easier to distinguish.

3. Within-subject word similarity
   - For one chosen subject, build per-word feature vectors and compute pairwise distances and similarity (1/(1+dist)).
   - Output: distance and similarity matrices (heatmaps). Lower distance / higher similarity means two words are produced similarly by that subject.

## Visualization / saved outputs

- Intersubject CV heatmap (saved to Graphs/)
- Fisher score barplot and Mahalanobis heatmap (saved to Graphs/)
- Within-subject similarity heatmap for a chosen subject (saved to Graphs/)
- Raw vs cleaned series comparison and cleaned correlation plots (saved to Graphs/)

Example saved filenames (created by test harness):

- Graphs/Amish_refined_comparison.png
- Graphs/Amish_refined_correlation.png
- Graphs/word_separability_fisher_score.png
- Graphs/word_separability_distance_matrix.png

## Quick run examples

- Gather summaries for words and subjects:
  - from eda_src.data_loader import gather_motion_summaries
  - summaries, errors = gather_motion_summaries(words, names, base_path="Data/recordings")
- Produce full reports and plots:

  - from eda_src.analysis import compute_word_similarity_for_subject, intersubject_variability_table, word_separability_metrics
  - from eda_src.visualization import plot_intersubject_variability, plot_word_separability, plot_within_subject_similarity
  - results = produce_reports(words, names, base_path="Data/recordings", subject_for_word_similarity="Bansbir")

- Run the preprocessing + plotting test (example path expected in code):
  - python Preprocessing/main.py
  - (This runs automated_preprocess on the example CSV and saves comparison plots in Graphs/)

## Short summary of results (rough / qualitative)

- Preprocessing: pipeline successfully fills short gaps and replaces severe outliers using smoothing + robust MAD; boolean masks track changes.
- Intersubject variability: some metrics show high CV across subjects (indicates inconsistent articulation / sensor placement). Focus on low-CV features for robust word models.
- Feature importance: Fisher scores highlight which metrics (e.g., range_theta vs x_disp) separate words best. Use highest Fisher features for simple classification experiments.
- Multivariate separation: Mahalanobis distances between word centroids give a compact measure of pairwise distinctiveness. Pairs with small distances are likely to be confusable.

## Limitations and next steps

* Assuming that the dataset is gaussian distributed, we are using Mahalanobis distance and Fisher score to measure the separability between words. 
*

# End of rough EDA summary
