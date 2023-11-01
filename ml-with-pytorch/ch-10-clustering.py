from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List
import random
import numpy as np
# k-means

X, y = make_blobs(n_samples=150, n_features=2, centers=3,
                  cluster_std=.5, shuffle=True, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolors='black', s=50)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.grid()
# plt.show()

km = KMeans(n_clusters=3, init='random', n_init=10,
            max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)
# plt.scatter(X[y_km == 0, 0],
#             X[y_km == 0, 1],
#             s=50, c='lightgreen',
#             marker='s', edgecolor='black',
#             label='Cluster 1')
# plt.scatter(X[y_km == 1, 0],
#             X[y_km == 1, 1],
#             s=50, c='orange',
#             marker='o', edgecolor='black',
#             label='Cluster 2')
# plt.scatter(X[y_km == 2, 0],
#             X[y_km == 2, 1],
#             s=50, c='lightblue',
#             marker='v', edgecolor='black',
#             label='Cluster 3')
# plt.scatter(km.cluster_centers_[:, 0],
#             km.cluster_centers_[:, 1],
#             s=250, marker='*',
#             c='red', edgecolor='black',
#             label='Centroids')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.tight_layout()
# plt.show()

# geos attempt


class GeoKMeans:
    def __init__(self, n_clusters=3, max_iter=20):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    # 1. Randomly pick k centroids from the examples as initial cluster centers
    # 2. Assign each example to the nearest centroid, 𝜇𝜇(𝑗𝑗), 𝑗𝑗 𝑗 {1, … , 𝑘𝑘}
    # 3. Move the centroids to the center of the examples that were assigned to it
    # 4. Repeat steps 2 and 3 until the cluster assignments do not change or a user-defined tolerance or maximum number of iterations is reached
    def fit(self, X: List):
        if len(X) < self.n_clusters:
            return X
        for index, centroid in enumerate(random.sample(X.tolist(), self.n_clusters)):
            self.centroids.append({'center': np.array(centroid), 'cluster': []})

        
        #implement the iterative step to improve the results
        #i'm pretty sure this is not right, but it works
        for i in range(self.max_iter):
            for index, x in enumerate(X):
                min_d = float("inf")
                my_centroid = None
                for centroid in self.centroids:
                    d = self.distance(x, centroid['center'])
                    if d < min_d:
                        my_centroid = centroid
                        min_d = d
                my_centroid['cluster'].append(x)

            for centroid in self.centroids:
                min_d = float("inf")
                new_centroid = centroid
                for member in centroid['cluster']:
                    total_distance = 0
                    for memeber2 in centroid['cluster']:
                        total_distance += self.distance(member, memeber2)
                    if min_d > total_distance:
                        new_centroid = member
                centroid['cluster'] = []
                centroid['center'] = new_centroid


                    
        for index, x in enumerate(X):
                min_d = float("inf")
                my_centroid = None
                for centroid in self.centroids:
                    d = self.distance(x, centroid['center'])
                    if d < min_d:
                        my_centroid = centroid
                        min_d = d
                my_centroid['cluster'].append(x)

        results = []
        for index, x in enumerate(X):
            label =  -1
            for l, c in enumerate(self.centroids):
                if x in np.array(c['cluster']):
                    label = l
                    break
            results.append(label)

        return np.array(results)

    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

gkme = GeoKMeans()
y_km = gkme.fit(X)
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
center = np.array([d['center'] for d in gkme.centroids])
plt.scatter(center[:, 0],
            center[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()
