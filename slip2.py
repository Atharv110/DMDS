import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generating a synthetic dataset
n_samples = 300
centers = 4
X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)

# Applying k-means algorithm
kmeans = KMeans(n_clusters=centers, random_state=42)
kmeans.fit(X)

# Getting cluster centers and labels
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# Visualizing the clusters and cluster centers
plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7, edgecolors='k')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='*', label='Cluster Centers')
plt.title('K-Means Clustering')
plt.legend()
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)
plt.show()
