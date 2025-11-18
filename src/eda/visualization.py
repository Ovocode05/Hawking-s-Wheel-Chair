import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # Added to help create save paths

def plot_intersubject_variability(intersubj_table, save_path=None):
    """
    Visualizes the Coefficient of Variation (CV) for intersubject variability.
    """
    if intersubj_table.empty:
        print("Intersubject table is empty, skipping plot.")
        return
    try:
        cv_data = intersubj_table.xs('cv', level=1, axis=1)
    except KeyError:
        print("Could not find 'cv' data in intersubject table, skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cv_data, 
        annot=True, 
        fmt=".2f", 
        cmap="viridis",
        linewidths=.5
    )
    plt.title("Intersubject Variability (Coefficient of Variation)\nHigher Value = More Variation Between People", fontsize=14)
    plt.xlabel("Motion Feature")
    plt.ylabel("Word")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Intersubject Variability plot to {save_path}")
        plt.close() # Close the plot after saving

def plot_word_separability(separability_results, save_dir=None):
    """
    Visualizes Fisher Score and Mahalanobis Distance.
    Saves two separate files to `save_dir`.
    """
    if not separability_results:
        print("Separability results are empty, skipping plots.")
        return

    # --- Part 1: Fisher Score Bar Plot ---
    fisher_df = separability_results.get("fisher_df")
    if fisher_df is not None and not fisher_df.empty:
        plt.figure(figsize=(10, 5))
        fisher_df_sorted = fisher_df.sort_values("fisher_score", ascending=False)
        sns.barplot(
            x=fisher_df_sorted["fisher_score"], 
            y=fisher_df_sorted.index,
            palette="rocket"
        )
        plt.title("Feature Importance (Fisher Score)\nHigher Score = Better at Separating Words", fontsize=14)
        plt.xlabel("Fisher Score")
        plt.ylabel("Motion Feature")
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, "word_separability_fisher_score.png")
            plt.savefig(save_path)
            print(f"Saved Fisher Score plot to {save_path}")
            plt.close()

    # --- Part 2: Mahalanobis Distance Heatmap ---
    dist_matrix = separability_results.get("pairwise_mahalanobis")
    if dist_matrix is not None and not dist_matrix.empty:
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(dist_matrix, dtype=bool))
        sns.heatmap(
            dist_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="mako_r", 
            mask=mask,
            linewidths=.5
        )
        plt.title("Inter-Word Distinction (Mahalanobis Distance)\nHigher Value = Words are More Different", fontsize=14)
        plt.xlabel("Word")
        plt.ylabel("Word")
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, "word_separability_distance_matrix.png")
            plt.savefig(save_path)
            print(f"Saved Distance Matrix plot to {save_path}")
            plt.close()

def plot_within_subject_similarity(summaries, subject_name, compute_similarity_func, save_path=None):
    """
    Visualizes word similarity for a single subject.
    'compute_similarity_func' is passed in from analysis.py
    """
    if not summaries:
        print("Summaries are empty, skipping within-subject plot.")
        return
        
    try:
        sim_results = compute_similarity_func(summaries, subject_name)
        sim_matrix = sim_results.get("similarity")
        
        if sim_matrix is None or sim_matrix.empty:
             print(f"No similarity matrix for {subject_name}, skipping plot.")
             return

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool))
        sns.heatmap(
            sim_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="mako",
            mask=mask,
            linewidths=.5,
            vmin=0, vmax=1
        )
        plt.title(f"Within-Subject Word SIMILARITY for '{subject_name}'\nHigher Value (Darker) = More Similar", fontsize=14)
        plt.xlabel("Word")
        plt.ylabel("Word")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved Within-Subject Similarity plot to {save_path}")
            plt.close()
            
    except Exception as e:
        print(f"Could not generate within-subject plot for '{subject_name}': {e}")