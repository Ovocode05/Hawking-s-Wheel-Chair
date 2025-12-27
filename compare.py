import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# ==========================================
def load_segments(path):
    data = np.load(path, allow_pickle=True)

    if "segments" not in data:
        raise KeyError("The NPZ file must contain key 'segments'")

    segs_raw = data["segments"]

    cleaned = []
    for seg in segs_raw:
        arr = np.array(seg)
        # Convert object → float
        if arr.dtype == object:
            arr = np.array(arr, dtype=float)
        cleaned.append(arr)

    print(f"Loaded {len(cleaned)} segments from {path}")
    return cleaned


# ==========================================
# DTW DISTANCE BETWEEN TWO SEGMENTS
# ==========================================
def dtw_dist(segA, segB):
    dist, _ = fastdtw(segA, segB, dist=euclidean)
    return dist


# ==========================================
# WORD-LEVEL DISTANCE + SIMILARITY
# ==========================================
def word_distance(wordA_segments, wordB_segments, nameA="WordA", nameB="WordB"):

    distances = []

    print(f"\n🔵 Computing similarity between '{nameA}' and '{nameB}' ...")

    # Compare every segment of word A with every segment of word B
    for i, segA in enumerate(wordA_segments):
        for j, segB in enumerate(wordB_segments):
            d = dtw_dist(segA, segB)
            distances.append(d)
            print(f"  DTW({nameA}_seg{i}, {nameB}_seg{j}) = {d:.3f}")

    distances = np.array(distances)
    final_distance = distances.mean()         # word-level distance
    final_similarity = 1 / (1 + final_distance)

    print("\n====================================")
    print(f" FINAL DISTANCE     ({nameA} vs {nameB}): {final_distance:.4f}")
    print(f" FINAL SIMILARITY   ({nameA} vs {nameB}): {final_similarity:.4f}")
    print("====================================\n")

    # OPTIONAL VISUALIZATION: overlay of representative segments (theta)
    plot_word_overlay(wordA_segments[0], wordB_segments[0], nameA, nameB)

    return final_distance, final_similarity


# ==========================================
# SIMPLE OVERLAY PLOT (theta only)
# ==========================================
def plot_word_overlay(segA, segB, nameA, nameB):
    plt.figure(figsize=(10,4))
    plt.plot(segA[:,0], label=f"{nameA} – θ")
    plt.plot(segB[:,0], label=f"{nameB} – θ")
    plt.title(f"Representative Overlay: {nameA} vs {nameB}")
    plt.xlabel("Frame Index")
    plt.ylabel("Theta")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# RUN FOR TWO WORDS
# ==========================================
bahaar = load_segments("d10bahaar.npz")
dard = load_segments("d10dard.npz")

word_distance(bahaar, dard, nameA="bahaar", nameB="dard")