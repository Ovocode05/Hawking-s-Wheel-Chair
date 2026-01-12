import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Words from your analysis
words = ['Bahaar', 'Bhuk', 'Doctor', 'Saah', 'Neend']

# Extracting the JS Divergence values from your heatmap (Results 1)
# Note: Values taken directly from image_bac047.png
js_matrix = np.array([
    [0.00, 0.28, 0.26, 0.24, 0.25], # Bahaar
    [0.28, 0.00, 0.25, 0.27, 0.28], # Bhuk
    [0.26, 0.25, 0.00, 0.30, 0.31], # Doctor
    [0.24, 0.27, 0.30, 0.00, 0.27], # Saah
    [0.25, 0.28, 0.31, 0.27, 0.00]  # Neend
])

# Generate the Linkage Matrix using 'ward' or 'complete' method
linked = linkage(js_matrix, 'complete')

plt.figure(figsize=(8, 5))
dendrogram(linked, labels=words, orientation='top', distance_sort='descending')
plt.title('Word Similarity Cluster (JS Divergence)')
plt.ylabel('Distance')
plt.show()