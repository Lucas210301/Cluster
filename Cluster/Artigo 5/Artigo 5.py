import numpy as np
from sklearn.cluster import KMeans

# Dados e partições
n = 4
partitions = [
    [[0, 1], [2, 3]],
    [[0, 2], [1, 3]]
]

# Construir matriz de similaridade
S = np.zeros((n, n))
for partition in partitions:
    for cluster in partition:
        for i in cluster:
            for j in cluster:
                S[i, j] += 1
S /= len(partitions)

# Aplicar k-means na matriz de similaridade
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(S)
print("Clusters finais:", labels)
