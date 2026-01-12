
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Configuration
DATA_ROOT = r'c:\Users\HP\GitMakesMeHappy\Hawking-s-Wheel-Chair\Data\recordings'
# Corrected OUTPUT_ROOT to avoid mixing graphs with data recordings
OUTPUT_ROOT = r'c:\Users\HP\GitMakesMeHappy\Hawking-s-Wheel-Chair\Data\Graphs'
APPS_WORDS = ['Bahaar','Bhuk','Doctor','Saah','Neend']
APPS_SUBJECTS = ['Amish', 'anubhavjot', 'Jaskaran', "Armman"]

WORDS = APPS_WORDS
SUBJECTS = APPS_SUBJECTS

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def load_data(word, subject):
    """Loads CSV data for a given word and subject."""
    path = os.path.join(DATA_ROOT, word, f"{subject}.csv")
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    
    try:
        df = pd.read_csv(path)
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            print(f"Warning: No numeric data in {path}")
            return None
        return df_numeric.values
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def dtw_distance(s1, s2):
    """Computes DTW distance."""
    if s1 is None or s2 is None: return np.inf
    # Optimization: Downsample if too large? 
    # For now, keep full resolution.
    cost_matrix = cdist(s1, s2, metric='euclidean')
    n, m = cost_matrix.shape
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 0] = 0
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m]

def js_divergence(s1, s2):
    """Computes Jensen-Shannon divergence."""
    if s1 is None or s2 is None: return np.inf
    d = s1.shape[1]
    if s2.shape[1] != d: return np.inf
    total_js = 0
    valid_dims = 0
    for k in range(d):
        data1, data2 = s1[:, k], s2[:, k]
        min_val, max_val = min(data1.min(), data2.min()), max(data1.max(), data2.max())
        if max_val == min_val: continue
        bins = np.linspace(min_val, max_val, 50)
        p, _ = np.histogram(data1, bins=bins, density=True)
        q, _ = np.histogram(data2, bins=bins, density=True)
        p, q = p + 1e-10, q + 1e-10
        p, q = p / p.sum(), q / q.sum()
        js = jensenshannon(p, q)
        if not np.isnan(js):
            total_js += js
            valid_dims += 1
    return total_js / valid_dims if valid_dims > 0 else 0.0

def plot_boxplot(data_dict, title, filename, ylabel="Distance"):
    """
    Plots a boxplot from a dictionary where keys are x-labels and values are lists of numbers.
    uses a long-form DataFrame to avoid seaborn issues.
    """
    
    # Filter out empty entries and create separate lists
    records = []
    for label, metrics in data_dict.items():
        if metrics and len(metrics) > 0:
            for val in metrics:
                # Ensure validation
                if np.isfinite(val):
                    records.append({'Label': label, 'Value': val})
    
    if not records:
        print(f"Warning: No valid data to plot for {title}")
        return

    df_plot = pd.DataFrame(records)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_plot, x='Label', y='Value')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Words")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_ROOT, filename)
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_heatmap_avg(matrix, labels, title, filename):
    plt.figure(figsize=(8, 6))
    if np.all(matrix == 0):
        print(f"Warning: Matrix for {title} is all zeros.")
    
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="viridis")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_ROOT, filename)
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_comparison_distribution(inter_subject, inter_word, title, filename):
    plt.figure(figsize=(8, 6))
    if not inter_subject or not inter_word:
        print(f"Warning: Insufficient data for distribution plot {title}")
        return

    sns.kdeplot(inter_subject, fill=True, label='Inter-Subject (Same Word)')
    sns.kdeplot(inter_word, fill=True, label='Inter-Word (Same Subject)')
    plt.title(title)
    plt.xlabel('Distance')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_ROOT, filename)
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def main():
    print("Loading data...")
    print(f"Words: {WORDS}")
    print(f"Subjects: {SUBJECTS}")
    
    data = {w: {s: load_data(w, s) for s in SUBJECTS} for w in WORDS}
    
    # 1. Inter-Subject Variability (Per Word)
    dtw_inter_subject = {w: [] for w in WORDS}
    js_inter_subject = {w: [] for w in WORDS}
    
    all_is_dtw = [] 
    all_is_js = []
    
    print("Computing Inter-Subject Variability...")
    for word in WORDS:
        subjs = [s for s in SUBJECTS if data[word][s] is not None]
        if len(subjs) < 2:
            print(f"  Skipping {word} (insufficient subjects: {len(subjs)})")
            continue
            
        for s1, s2 in combinations(subjs, 2):
            d1, d2 = data[word][s1], data[word][s2]
            
            dtw = dtw_distance(d1, d2)
            js = js_divergence(d1, d2)
            
            dtw_inter_subject[word].append(dtw)
            js_inter_subject[word].append(js)
            all_is_dtw.append(dtw)
            all_is_js.append(js)

    plot_boxplot(dtw_inter_subject, "DTW Inter-Subject Variability", "dtw_inter_subject_boxplot.png")
    plot_boxplot(js_inter_subject, "JS Inter-Subject Variability", "js_inter_subject_boxplot.png", ylabel="Divergence")

    # 2. Inter-Word Distinction
    n_words = len(WORDS)
    dtw_sum_mat = np.zeros((n_words, n_words))
    js_sum_mat = np.zeros((n_words, n_words))
    valid_counts = np.zeros((n_words, n_words))
    
    all_iw_dtw = []
    all_iw_js = []
    
    print("Computing Inter-Word Distinction...")
    for s in SUBJECTS:
        print(f"  Processing Subject: {s}")
        for i, w1 in enumerate(WORDS):
            d1 = data[w1][s]
            if d1 is None: continue
            
            for j, w2 in enumerate(WORDS):
                if i == j: continue 
                
                d2 = data[w2][s]
                if d2 is None: continue
                
                dtw = dtw_distance(d1, d2)
                js = js_divergence(d1, d2)
                
                dtw_sum_mat[i, j] += dtw
                js_sum_mat[i, j] += js
                valid_counts[i, j] += 1
                
                if i < j:
                    all_iw_dtw.append(dtw)
                    all_iw_js.append(js)

    valid_counts[valid_counts == 0] = 1
    dtw_avg_mat = dtw_sum_mat / valid_counts
    js_avg_mat = js_sum_mat / valid_counts
    
    plot_heatmap_avg(dtw_avg_mat, WORDS, "Average DTW Inter-Word Distance", "dtw_inter_word_avg_heatmap.png")
    plot_heatmap_avg(js_avg_mat, WORDS, "Average JS Inter-Word Divergence", "js_inter_word_avg_heatmap.png")
    
    # 3. Global Comparison
    plot_comparison_distribution(all_is_dtw, all_iw_dtw, "DTW Separability", "dtw_separability.png")
    plot_comparison_distribution(all_is_js, all_iw_js, "JS Separability", "js_separability.png")

    print("Analysis Complete.")

if __name__ == "__main__":
    main()
