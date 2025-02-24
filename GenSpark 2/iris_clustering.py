import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load the iris dataset
data = pd.read_csv('C:/Python Codes/GenSpark 2/iris_data.csv')
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Separate features (X) and target (y)
X = data.iloc[:, 0:4]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# KMeans clustering
kmeans = KMeans(n_clusters = 3, random_state = 2025)
kmeans.fit(X)

# Hierarchical clustering
hier = AgglomerativeClustering(n_clusters = 3)
hier.fit(X)

# Evaluate KMeans clustering
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X, kmeans.labels_)

# Evaluate Hierarchical clustering
hier_silhouette = silhouette_score(X, hier.labels_)

print(f"KMeans Inertia: {kmeans_inertia}")
print(f"KMeans Silhouette Score: {kmeans_silhouette}")
print(f"Hierarchical Silhouette Score: {hier_silhouette}")

# Experiment with different numbers of clusters for K-Means
inertia = []
silhouette = []
for n_clusters in range(2, 11):
    Tkmeans = KMeans(n_clusters = n_clusters, random_state = 2025, n_init = 'auto')
    Tkmeans.fit(X)
    inertia.append(Tkmeans.inertia_)
    silhouette.append(silhouette_score(X, Tkmeans.labels_))

    print(f"For n_clusters = {n_clusters}, inertia is {Tkmeans.inertia_} and silhouette score is {silhouette_score(X, Tkmeans.labels_)}")

# PCA for visualization
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

# Create a DataFrame for the PCA components
pca_df = pd.DataFrame(data = X_pca, columns = ['PCA1', 'PCA2'])

# KMeans clustering plot
plt.figure(figsize = (8, 6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c = kmeans.labels_, cmap = 'viridis')
plt.title('KMeans Clustering with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label = 'Cluster')
plt.show()

# Hierarchical clustering plot
plt.figure(figsize = (8, 6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c = hier.labels_, cmap = 'viridis')
plt.title('Hierarchical Clustering with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label = 'Cluster')
plt.show()

# Map original labels to numeric values
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data['class_numeric'] = data['class'].map(class_mapping)
y_true = data['class_numeric']

# Evaluate KMeans clustering
kmeans_ari = adjusted_rand_score(y_true, kmeans.labels_)
kmeans_confusion = confusion_matrix(y_true, kmeans.labels_)

# Evaluate Hierarchical clustering
hier_ari = adjusted_rand_score(y_true, hier.labels_)
hier_confusion = confusion_matrix(y_true, hier.labels_)

print(f"KMeans Adjusted Rand Index: {kmeans_ari}")
print(f"KMeans Confusion Matrix:\n{kmeans_confusion}")
print(f"Hierarchical Adjusted Rand Index: {hier_ari}")
print(f"Hierarchical Confusion Matrix:\n{hier_confusion}")

# Define a function to plot decision boundaries
def plot_decision_boundaries(X, labels, title):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired, aspect='auto', origin='lower')

    plt.plot(X.iloc[:, 0], X.iloc[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = fitted_model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

# KMeans decision boundary plot
kmeans_pca = KMeans(n_clusters = 3, random_state = 2025, n_init = 'auto')
fitted_model = kmeans_pca.fit(X_pca)
plot_decision_boundaries(pca_df, kmeans_pca.labels_, 'KMeans Decision Boundaries with PCA')

# Hierarchical decision boundary plot
hier_pca = AgglomerativeClustering(n_clusters = 3)
fitted_model = hier_pca.fit(X_pca)
plot_decision_boundaries(pca_df, hier_pca.labels_, 'Hierarchical Decision Boundaries with PCA')