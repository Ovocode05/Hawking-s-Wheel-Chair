import os
import matplotlib.pyplot as plt

# Import all our custom modules from the 'src' folder
from eda.data_loader import gather_motion_summaries
from eda.analysis import (
    intersubject_variability_table, 
    word_separability_metrics,
    compute_word_similarity_for_subject
)
from eda.visualization import (
    plot_intersubject_variability, 
    plot_word_separability,
    plot_within_subject_similarity
)

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- 1. Configuration ---
    WORDS = ['Bahaar','Bhuk','Doctor','Saah','Neend']
    NAMES = ['Armman','Bansbir','Amish']
    DATA_PATH = "Data/recordings"
    RESULTS_DIR = "results"
    
    # Create the results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 2. Data Loading and Feature Extraction ---
    print("Gathering and processing data...")
    summaries, errors = gather_motion_summaries(
        WORDS, 
        NAMES, 
        base_path=DATA_PATH,
        columns=["theta", "x", "y"] # Specify features to use
    )
    
    if not summaries or all(not v for v in summaries.values()):
        print("No data was successfully processed. Exiting.")
        if errors:
            print("\nErrors encountered:")
            for e in errors: print(e)
        exit()

    # --- 3. Statistical Analysis ---
    print("Running statistical analysis...")
    # Intersubject variability
    intersubj_table = intersubject_variability_table(summaries)
    
    # Inter-word separability
    separability_results = word_separability_metrics(summaries)

    # --- 4. Console Reports ---
    print("\n--- Intersubject variability (CV) ---")
    if not intersubj_table.empty:
        try:
            # Try to extract just the 'cv' column for a cleaner print
            print(intersubj_table.xs('cv', level=1, axis=1).round(4))
        except KeyError:
            print(intersubj_table.round(4)) # Fallback to printing all
    
    print("\n--- Per-metric Fisher scores (Higher is Better) ---")
    if separability_results.get("fisher_df") is not None:
        print(separability_results["fisher_df"].sort_values("fisher_score", ascending=False).round(4))

    print("\n--- Pairwise Mahalanobis distances (Higher is More Distinct) ---")
    if separability_results.get("pairwise_mahalanobis") is not None:
        print(separability_results["pairwise_mahalanobis"].round(3))
        
    if errors:
        print("\n--- Warnings / Errors ---")
        for e in errors: print(e)

    # --- 5. Generate and Save Visualizations ---
    print("\nGenerating and saving plots...")
    
    # Q1: Intersubject Variability
    plot_intersubject_variability(
        intersubj_table, 
        save_path=os.path.join(RESULTS_DIR, "1_intersubject_variability_cv.png")
    )
    
    # Q2: Inter-word Distinction
    plot_word_separability(
        separability_results, 
        save_dir=RESULTS_DIR
    )
    
    # Bonus: Within-Subject Similarity (for the first person)
    if NAMES:
        subject_name = NAMES[2]
        plot_within_subject_similarity(
            summaries, 
            subject_name,
            compute_word_similarity_for_subject, # Pass the analysis function
            save_path=os.path.join(RESULTS_DIR, f"3_similarity_for_{subject_name}.png")
        )

    print(f"\nAll plots saved to '{RESULTS_DIR}' folder.")
    