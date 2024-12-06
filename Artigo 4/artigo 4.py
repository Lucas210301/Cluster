from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Dados de exemplo
X = [[1, 1], [2, 2], [1, 2], [10, 10], [11, 11], [10, 11]]

# Aplicar k-means
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)

# Calcular índice de silhueta
silhouette = silhouette_score(X, labels)
print(f"Índice de Silhueta: {silhouette}")
