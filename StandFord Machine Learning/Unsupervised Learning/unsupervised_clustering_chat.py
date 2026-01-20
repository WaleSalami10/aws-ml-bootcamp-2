import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data (Same as before for consistency)
X, y_true = make_blobs(n_samples=200, centers=[(40, 15), (65, 30)], cluster_std=[8, 8], random_state=42)

# Rescale and clip the feature values to a more realistic range
X[:, 0] = np.clip(X[:, 0], 20, 85) # Age: 20 to 85 years
X[:, 1] = np.clip(X[:, 1], 5, 55)  # Tumor Size: 5mm to 55mm

data = pd.DataFrame(X, columns=['Age', 'Tumor Size (mm)'])

# 2. Apply an Unsupervised Learning Algorithm (K-Means Clustering)
# We'll assume we want to find 2 clusters, similar to the supervised case
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Add the cluster labels to the dataframe
data['Cluster'] = y_kmeans

# 3. Plot the Unsupervised Clusters
plt.figure(figsize=(10, 7))

# Scatter plot of the data points, colored by their assigned cluster
scatter = plt.scatter(data['Age'], data['Tumor Size (mm)'], c=data['Cluster'],
                      cmap='viridis', edgecolors='k', s=80, alpha=0.8)

# Plot the cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Cluster Centroids', edgecolors='k')

# Create a legend for the clusters
legend1 = plt.legend(*scatter.legend_elements(),
                     loc='lower right', title='Cluster',
                     labels=[f'Cluster {i}' for i in range(2)])
plt.gca().add_artist(legend1)

plt.legend(loc='upper left') # Legend for centroids

plt.xlabel('Age (Years)', fontsize=14)
plt.ylabel('Tumor Size (mm)', fontsize=14)
plt.title('Unsupervised Clustering of Benign and Malignant Tumors (K-Means)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('unsupervised_clustering_chart.png')
print("unsupervised_clustering_chart.png")