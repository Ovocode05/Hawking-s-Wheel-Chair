import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Conv1D, GlobalAveragePooling1D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# ======================================================
# FIX RANDOMNESS
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================================================
# CV-OPTIMIZED CONFIG
# ======================================================
DATA_DIR = r"D:\Desktop\Research2\segmented_data_trial"
OUT_DIR  = "outputs1_cv"

MAX_LEN = 120
FEATURES = 5   # CV features (landmarks, lip positions, etc.)
LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS = 60
MARGIN = 0.8   # Tighter margin for CV features

# ======================================================
# CREATE OUTPUT DIRS
# ======================================================
os.makedirs(OUT_DIR, exist_ok=True)
for d in ["embeddings", "similarity", "clustering", "metadata"]:
    os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

# ======================================================
# LOAD CV DATA
# ======================================================
def load_all_words(data_dir):
    X, y = [], []
    label_map = {}
    label_id = 0

    for subject in os.listdir(data_dir):
        sp = os.path.join(data_dir, subject)
        if not os.path.isdir(sp): continue

        for file in os.listdir(sp):
            if not file.endswith(".npz"): continue

            word = file.replace(".npz", "").lower()
            if word not in label_map:
                label_map[word] = label_id
                label_id += 1

            data = np.load(os.path.join(sp, file), allow_pickle=True)
            if "segments" not in data: continue

            for seg in data["segments"]:
                if seg is not None and len(seg) > 0:
                    X.append(seg)
                    y.append(label_map[word])

    return X, np.array(y), label_map

X_raw, y, label_map = load_all_words(DATA_DIR)

if len(X_raw) == 0:
    raise RuntimeError("❌ No CV segments loaded.")

pd.DataFrame(label_map.items(), columns=["word", "label"]).to_csv(
    os.path.join(OUT_DIR, "metadata", "label_map.csv"), index=False
)
print(f"✅ Loaded {len(X_raw)} CV segments, {len(label_map)} words")

# ======================================================
# CV-SPECIFIC PREPROCESSING (ROBUST TO OUTLIERS)
# ======================================================
def pad_segments(segs, max_len):
    out = []
    for s in segs:
        if len(s) > max_len:
            out.append(s[:max_len])
        else:
            out.append(np.pad(s, ((0, max_len - len(s)), (0, 0))))
    return np.array(out)

X = pad_segments(X_raw, MAX_LEN)

# RobustScaler better for CV landmark data (handles outliers)
scaler = RobustScaler()
X = scaler.fit_transform(X.reshape(-1, FEATURES)).reshape(X.shape)

# ======================================================
# CV-OPTIMIZED ENCODER (1D CNN + LSTM)
# ======================================================
def build_cv_encoder():
    inp = Input(shape=(MAX_LEN, FEATURES))
    
    # 1D CNN for local spatial patterns (lip movements)
    x = Conv1D(128, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # LSTM for temporal dynamics
    x = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    
    # Final projection
    x = Dense(LATENT_DIM, activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(LATENT_DIM)(x)
    
    return Model(inp, out)

encoder = build_cv_encoder()
encoder.summary()

# ======================================================
# SIAMESE NETWORK (UNCHANGED)
# ======================================================
a = Input(shape=(MAX_LEN, FEATURES))
b = Input(shape=(MAX_LEN, FEATURES))
ea, eb = encoder(a), encoder(b)

def squared_euclidean(v):
    x, y = v
    return tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)

dist = Lambda(squared_euclidean)([ea, eb])
siamese = Model([a, b], dist)

def contrastive_loss(y_true, d):
    y_true = tf.cast(y_true, tf.float32)
    pos = y_true * d
    neg = (1 - y_true) * tf.maximum(MARGIN - d, 0.0)
    return tf.reduce_mean(pos + neg)

siamese.compile(optimizer=Adam(1e-3), loss=contrastive_loss)

# ======================================================
# BALANCED PAIR GENERATOR
# ======================================================
def pair_generator(X, y, batch):
    n = len(X)
    label_indices = {lbl: np.where(y == lbl)[0] for lbl in np.unique(y)}
    
    while True:
        Xa, Xb, lab = [], [], []
        for _ in range(batch):
            if np.random.rand() < 0.5:  # Positive pair
                lbl = np.random.choice(list(label_indices.keys()))
                i, j = np.random.choice(label_indices[lbl], 2, replace=False)
                Xa.append(X[i]); Xb.append(X[j]); lab.append(1)
            else:  # Negative pair
                lbl1, lbl2 = np.random.choice(list(label_indices.keys()), 2, replace=False)
                i = np.random.choice(label_indices[lbl1])
                j = np.random.choice(label_indices[lbl2])
                Xa.append(X[i]); Xb.append(X[j]); lab.append(0)
        yield (np.array(Xa), np.array(Xb)), np.array(lab)

# ======================================================
# TRAIN
# ======================================================
print("🚀 Training CV Siamese Network...")
history = siamese.fit(
    pair_generator(X, y, BATCH_SIZE),
    steps_per_epoch=len(X) // BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

encoder.save(os.path.join(OUT_DIR, "cv_siamese_encoder.keras"))
print("✅ CV encoder saved")

# ======================================================
# WORD EMBEDDINGS
# ======================================================
label_to_word = {v: k for k, v in label_map.items()}
embeddings = encoder.predict(X, batch_size=32, verbose=1)

words, word_embs = [], []
for lbl in sorted(label_to_word):
    word_embs.append(np.mean(embeddings[y == lbl], axis=0))
    words.append(label_to_word[lbl])

word_embs = np.array(word_embs)

# Save
pd.DataFrame(word_embs, index=words).to_csv(
    os.path.join(OUT_DIR, "embeddings", "cv_word_embeddings.csv")
)
np.save(os.path.join(OUT_DIR, "embeddings", "cv_word_embeddings.npy"), word_embs)

# ======================================================
# CV-SPECIFIC 2D VISUALIZATION
# ======================================================
pca = PCA(n_components=2)
word_embs_2d = pca.fit_transform(word_embs)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(word_embs_2d[:, 0], word_embs_2d[:, 1], 
                     c=list(range(len(words))), cmap='tab20', s=200, alpha=0.8)
for i, word in enumerate(words):
    plt.annotate(word, (word_embs_2d[i, 0], word_embs_2d[i, 1]), 
                xytext=(8, 8), textcoords='offset points', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
plt.title("Silent Speech Word Embeddings (PCA 2D) - CV Features", fontsize=16)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "embeddings", "cv_embeddings_2d.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ======================================================
# DISTANCE MATRICES
# ======================================================
euclidean_dist = euclidean_distances(word_embs)  # Better for CV
cosine_sim = cosine_similarity(word_embs)

pd.DataFrame(euclidean_dist, index=words, columns=words).to_csv(
    os.path.join(OUT_DIR, "similarity", "cv_euclidean_distance.csv")
)

# ======================================================
# CV-OPTIMIZED HIERARCHICAL CLUSTERING
# ======================================================
linkage_methods = ['ward', 'average']  # Ward excellent for CV embeddings
truncate_mode = 'level'
p = max(12, len(words) // 1.5)

for method in linkage_methods:
    print(f"🔗 CV Clustering with {method}...")
    
    if method == 'ward':
        Z = linkage(word_embs, method=method)  # Direct on normalized embeddings
    else:
        Z = linkage(squareform(euclidean_dist), method=method)
    
    np.save(os.path.join(OUT_DIR, "clustering", f"cv_linkage_{method}.npy"), Z)
    
    # BEAUTIFUL DENDROGRAM FOR CV DATA
    plt.figure(figsize=(20, 12))
    dendrogram(
        Z, labels=words, leaf_rotation=90, leaf_font_size=11,
        truncate_mode=truncate_mode, p=p, show_contracted=True,
        above_threshold_color='gray',
        color_threshold=0.6 * Z[-1, 2],  # Show main clusters clearly
        orientation='top'
    )
    plt.title(f"Silent Speech CV Clustering - {method.title()} Linkage", fontsize=18)
    plt.ylabel("Euclidean Distance", fontsize=14)
    plt.xlabel("Word Labels", fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "clustering", f"cv_dendrogram_{method}.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ======================================================
# TOP SIMILAR WORDS (PHONETIC NEIGHBORS)
# ======================================================
sim_df = pd.DataFrame(cosine_sim, index=words, columns=words)
top_pairs = []
for i in range(len(words)):
    for j in range(i+1, len(words)):
        top_pairs.append((words[i], words[j], sim_df.iloc[i,j]))

top_pairs.sort(key=lambda x: x[2], reverse=True)
top_pairs_df = pd.DataFrame(top_pairs[:15], columns=['word1', 'word2', 'cosine_sim'])
top_pairs_df.to_csv(os.path.join(OUT_DIR, "similarity", "cv_top_similar_pairs.csv"), index=False)

print("\n🎯 TOP CV SIMILAR WORD PAIRS (VISUAL FEATURES):")
print(top_pairs_df.head(10).to_string(index=False))

print(f"\n✅ CV SILENT SPEECH ANALYSIS COMPLETE! Check {OUT_DIR}")
print("📁 BEST VISUALS:")
print("   • cv_embeddings_2d.png (MUST SEE)")
print("   • cv_dendrogram_ward.png (MAIN RESULT)")
print("   • cv_top_similar_pairs.csv")
