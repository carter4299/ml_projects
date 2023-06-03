"""
originally jupyter notebook
"""
import numpy as np
import pandas as pd
import seaborn as sns
from google.colab import drive
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

#drive.mount('/content/drive')
file_name = '/content/drive/MyDrive/Colab Notebooks/Wine_Quality_Data.csv'
with open(file_name, 'r') as file:
    df = pd.read_csv(file_name)

num_rows, num_columns = df.shape
#print(f"Dataset \"{file_name}\", has {num_rows} rows and {num_columns} columns.\n")

df.drop('color', axis=1, inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

pca = PCA()
pca_transformed_data = pca.fit_transform(scaled_features)

cumulative_sum = np.cumsum(pca.explained_variance_ratio_)
num_of = range(1, len(cumulative_sum) + 1)

plt.step(num_of, cumulative_sum)
plt.plot(num_of, cumulative_sum)

plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

n_components = np.argwhere(cumulative_sum >= 0.9)[0][0] + 1
print(f"\tIt takes {n_components} features to reach 90% variance.\n")
pca_new = PCA(n_components=n_components)
pca_transformed_data_new = pca_new.fit_transform(scaled_features)

#-------------------------------------------------------------

np.random.seed(42)
model = KMeans()
visualizer = KElbowVisualizer(model)
visualizer.fit(pca_transformed_data_new)

visualizer.show()

optimal_clust = visualizer.elbow_value_
print(f"\tThe optimal number of clusters is: {optimal_clust}\n")

kmeans_optimal = KMeans(n_clusters=optimal_clust, random_state=42)
kmeans_optimal.fit(pca_transformed_data_new)

labels = kmeans_optimal.labels_

silhouette_val= silhouette_score(pca_transformed_data_new, labels)
label_counts = pd.Series(labels).value_counts(normalize=True) * 100

print("Value counts of unique class labels in percentage:")
print(label_counts)
print(f"\nSilhouette score: {silhouette_val}\n")

#-------------------------------------------------------------

np.random.seed(42)

model2 = KMeans()
visualizer2 = KElbowVisualizer(model2)
visualizer2.fit(scaled_features)

visualizer2.show()

optimal_clust2 = visualizer2.elbow_value_
print(f"\tThe optimal number of clusters is: {optimal_clust2}\n")

kmeans_optimal2 = KMeans(n_clusters=optimal_clust2, random_state=42)
kmeans_optimal2.fit(scaled_features)

labels2 = kmeans_optimal2.labels_

silhouette_scores2 = silhouette_score(scaled_features, labels2)

label_counts2 = pd.Series(labels2).value_counts(normalize=True) * 100

print("Value counts of unique class labels in percentage (without PCA selection):")
print(label_counts2)
print(f"\nSilhouette score (without PCA selection): {silhouette_scores2}\n")

print(f"\t The silhouette score of the PCA selected data(~0.255) was ~0.024 higher than the data with no PCA selection ~0.231")


#-------------------------------------------------------------
linkage_matrix = linkage(scaled_features, method='ward')

plt.figure()
dendrogram(linkage_matrix)
plt.title('Dendrogram (Ward Method)')
plt.xlabel('Data points')
plt.ylabel('Euclidean distance')
plt.show()

best_k = 2

agg_clustering_best = AgglomerativeClustering(n_clusters=best_k, linkage='ward')

labels_agg_best = agg_clustering_best.fit_predict(scaled_features)

sil_score_agg_best = silhouette_score(scaled_features, labels_agg_best)

print(f"\nBest number of clusters (k): {best_k}")
print(f"Silhouette score for k={best_k}: {sil_score_agg_best}")
#-------------------------------------------------------------
nearest_neighbors = NearestNeighbors(n_neighbors=4)
nearest_neighbors.fit(scaled_features)

distances, indices = nearest_neighbors.kneighbors(scaled_features)
distances_4th_neighbor = distances[:, -1]
sorted_distances = np.sort(distances_4th_neighbor)

plt.figure()
plt.plot(sorted_distances)
plt.xlabel('Data points (sorted by distance)')
plt.ylabel('Distance of the 4th Nearest Neighbor')
plt.show()
print('\t~2.5 distance points\n')

dbscan_model = DBSCAN(eps=2.25, min_samples=10)

dbscan_model.fit(scaled_features)
dbscan_labels = dbscan_model.labels_

unique_labels, counts = np.unique(dbscan_labels, return_counts=True)
label_per = counts / len(dbscan_labels) * 100

print("Percentage of unique class labels:")
for label, per in zip(unique_labels, label_per):
    print(f"Label {label}: {per:.2f}%")

valid_l = dbscan_labels[dbscan_labels != -1]
valid_f = scaled_features[dbscan_labels != -1]
print(f"\nSilhouette score: {silhouette_score(valid_f, valid_l)}")

scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
scaled_df['cluster'] = dbscan_labels

sns.pairplot(scaled_df, hue='cluster', corner=True)
plt.show()
print("\tThe DBSCAN Clustering model had the highest silhouette score by a lot, which makes me think there could be an error.\n\tI tried messing around with different parameters and it still hovered around ~0.6.\t\n\tThe score ranking was:\n\t(1)DBSCAN Clustering = 0.637\n\t(2)KMeans Clustering w/ PCA selection = 0.255\n\t(3)Agglomerative Clustering = 0.250\n\t(4)KMeans Clustering w/o PCA selection = 0.231")
#-------------------------------------------------------------


















































