from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(iris_df.head())
# print(iris_df.isnull().sum())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_df)  # after fit_transform we get a numpy array, need to create a pd df again
iris_df_scaled = pd.DataFrame(iris_scaled, columns=iris.feature_names)
# print(iris_df_scaled.sample(10).round(2))

"""modeling"""
X = iris_df_scaled

from sklearn.cluster import KMeans
"""get the appropriate number of ckusters"""
wcss = []  # WCSS is the sum of the squared distance between each point and the centroid in a cluster - the mistake
for i in range (1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)   # we add the gotten intertia (mistake) to the list
#    print the wscc
plt.figure(figsize=(5, 5))
plt.plot(range(1, 11), wcss)
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()
#
#  #  from the graph looks like the inertia stops drastically changing after n_clusters = 3, so we choose 3

k_means = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
#   init defines the starting position of the centroids. if k-means++ they are located far away from each other
#   if random they are allocated randomly
#   max iter says how many times a centroid choice is made before stabilized, n_init says how often we do k-means++
y_pred = k_means.fit_predict(X)

"""evaluation"""
print(k_means.inertia_)  # check what wcss we got

plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='Paired')
"""plotting centroids. s = size, ^ = triangle"""
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1],
            s=150, marker='^', label='centroids', color='red')
plt.legend(loc='best')
plt.title('iris clustering')
plt.show()