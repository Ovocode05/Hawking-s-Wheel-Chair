import numpy as np
import pandas as pd
from itertools import combinations

def _sample_matrix_from_summaries(summaries, metric_keys=None):
    """
    Build sample matrix rows = (word,subject), cols = metric means.
    Helper function for other analysis functions.
    """
    X = []
    labels = []
    idx = []
    if metric_keys is None:
        for w in summaries:
            for s in summaries[w]:
                metric_keys = list(summaries[w][s].keys())
                if metric_keys: break
            if metric_keys: break
    
    if not metric_keys:
        return np.empty((0,0)), [], [], []

    for w, subj in summaries.items():
        for sname, summ in subj.items():
            vec = []
            for k in metric_keys:
                v = summ.get(k, {}).get("mean", np.nan)
                vec.append(float(v) if not pd.isna(v) else np.nan)
            
            vec = np.asarray(vec, dtype=float)
            if np.isnan(vec).all(): continue
                
            X.append(vec); labels.append(w); idx.append((w, sname))
            
    if len(X)==0:
        return np.empty((0,len(metric_keys))), [], [], metric_keys
        
    X = np.vstack(X)
    return X, labels, idx, metric_keys

def intersubject_variability_table(summaries, metric_list=None):
    """
    For each word and each metric compute CV.
    """
    rows = {}
    all_metrics = set()
    
    if metric_list is None:
        for w, subj_dict in summaries.items():
            for subj, summ in subj_dict.items():
                all_metrics.update(summ.keys())
        metric_list = sorted(list(all_metrics))
    
    for w, subj_dict in summaries.items():
        rows[w] = {}
        for m in metric_list:
            vals = [subj_dict.get(subj, {}).get(m, {}).get("mean", np.nan) for subj in subj_dict]
            arr = np.asarray(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            
            mean_of_means = float(np.mean(arr)) if arr.size>0 else np.nan
            std_of_means = float(np.std(arr, ddof=0)) if arr.size>0 else np.nan
            cv = float(std_of_means/mean_of_means) if (arr.size>0 and mean_of_means!=0) else np.nan
            
            rows[w][(m,"mean")] = mean_of_means
            rows[w][(m,"std")] = std_of_means
            rows[w][(m,"cv")] = cv
            
    if not rows: return pd.DataFrame()
        
    df = pd.DataFrame.from_dict(rows, orient='index')
    if df.empty: return pd.DataFrame()
    
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Ensure all columns are present
    all_cols = pd.MultiIndex.from_product([metric_list, ['mean', 'std', 'cv']])
    df = df.reindex(columns=all_cols)
    return df

def word_separability_metrics(summaries):
    """
    Compute Fisher score and Mahalanobis distances.
    """
    X, labels, idx, metrics = _sample_matrix_from_summaries(summaries)
    if X.size==0:
        return {"fisher_df": pd.DataFrame(), "pairwise_mahalanobis": pd.DataFrame(), "centroids": pd.DataFrame()}

    metrics = list(metrics)
    unique_labels = sorted(set(labels))
    overall_mean = np.nanmean(X, axis=0)
    overall_mean = np.nan_to_num(overall_mean)
    fisher_scores = {}

    for j, m in enumerate(metrics):
        between, within = 0.0, 0.0
        for lab in unique_labels:
            inds = [i for i, l in enumerate(labels) if l == lab]
            class_vals = X[inds, j]
            class_vals = class_vals[~np.isnan(class_vals)]
            if class_vals.size == 0: continue
            mu_i = float(np.mean(class_vals))
            n_i = class_vals.size
            between += n_i * (mu_i - overall_mean[j])**2
            within += n_i * (np.var(class_vals, ddof=0))
        fisher_scores[m] = float(between / within) if within > 0 else np.nan

    fisher_df = pd.DataFrame.from_dict(fisher_scores, orient="index", columns=["fisher_score"])

    centroids = {lab: np.nanmean(X[[i for i, l in enumerate(labels) if l == lab], :], axis=0) for lab in unique_labels}
    centroids_df = pd.DataFrame.from_dict(centroids, orient="index", columns=metrics)

    col_means_impute = np.nan_to_num(np.nanmean(X, axis=0))
    residuals = []
    for lab in unique_labels:
        inds = [i for i, l in enumerate(labels) if l == lab]
        vals = X[inds, :]
        mu = np.nan_to_num(centroids[lab], nan=col_means_impute)
        vals_imp = np.where(np.isnan(vals), col_means_impute, vals)
        residuals.append(vals_imp - mu)
        
    if residuals:
        S = np.cov(np.vstack(residuals), rowvar=False, bias=False)
    else:
        S = np.eye(len(metrics))
    
    S_inv = np.linalg.pinv(S + np.eye(S.shape[0]) * 1e-6)

    words = centroids_df.index.tolist()
    pairwise = pd.DataFrame(index=words, columns=words, dtype=float)
    for a, b in combinations(words, 2):
        mu_a = np.nan_to_num(centroids_df.loc[a].to_numpy(), nan=col_means_impute)
        mu_b = np.nan_to_num(centroids_df.loc[b].to_numpy(), nan=col_means_impute)
        diff = mu_a - mu_b
        dist = float(np.sqrt(diff.T.dot(S_inv).dot(diff)))
        pairwise.loc[a, b] = dist; pairwise.loc[b, a] = dist
        
    for w in words: pairwise.loc[w, w] = 0.0
    return {"fisher_df": fisher_df, "pairwise_mahalanobis": pairwise, "centroids": centroids_df}

def compute_word_similarity_for_subject(summaries, subject, metric_keys=None, scale=True):
    """
    For a single `subject`, compute pairwise word similarity.
    """
    rows = {}
    if metric_keys is None:
         for w, subj_dict in summaries.items():
            if subj_dict.get(subject):
                metric_keys = list(subj_dict[subject].keys())
                break
    if metric_keys is None:
         raise ValueError(f"No data for subject '{subject}' to infer metric keys.")

    for w, subj_dict in summaries.items():
        summ = subj_dict.get(subject)
        if not summ: continue
        vec, ok = [], True
        for k in metric_keys:
            v = summ.get(k, {}).get("mean", np.nan)
            if pd.isna(v): ok = False; break
            vec.append(float(v))
        if ok: rows[w] = np.asarray(vec, dtype=float)
            
    if not rows:
        raise ValueError(f"No complete data for subject '{subject}' across provided words.")

    words_list = list(rows.keys())
    M = np.vstack([rows[w] for w in words_list])
    
    if scale:
        mu = np.nanmean(M, axis=0)
        sigma = np.nanstd(M, axis=0, ddof=0)
        sigma[sigma == 0] = 1.0
        M = (M - mu) / sigma

    D = np.sqrt(((M[:, None, :] - M[None, :, :])**2).sum(axis=2))
    dist_df = pd.DataFrame(D, index=words_list, columns=words_list)
    sim_df = 1.0 / (1.0 + dist_df)
    return {"distances": dist_df, "similarity": sim_df, "metric_keys": metric_keys}