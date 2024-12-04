import numpy as np

# Matrizes de dissimilaridade
D1 = np.array([
    [0, 1, 2, 3],
    [1, 0, 2, 2],
    [2, 2, 0, 1],
    [3, 2, 1, 0]
])
D2 = np.array([
    [0, 2, 1, 4],
    [2, 0, 3, 3],
    [1, 3, 0, 2],
    [4, 3, 2, 0]
])

D_matrices = [D1, D2]

# Inicialização
n_objects = 4
n_clusters = 2
clusters = {1: [0, 1], 2: [2, 3]}  # Indices dos objetos em cada cluster
prototypes = {1: 0, 2: 3}          # Protótipos iniciais
p = len(D_matrices)  # Número de matrizes de dissimilaridade

def calculate_lambda(cluster, prototype, D_matrices):
    """Cálculo dos pesos lambda"""
    lambdas = []
    total_d = []
    for D in D_matrices:
        sum_d = sum(D[obj, prototype] for obj in cluster)
        total_d.append(sum_d)
    product = np.prod(total_d)
    for sum_d in total_d:
        lambdas.append((product ** (1 / p)) / sum_d)
    return lambdas

def update_clusters(D_matrices, lambdas, prototypes):
    """Atualização dos clusters baseada na dissimilaridade ajustada"""
    n_objects = D_matrices[0].shape[0]
    new_clusters = {1: [], 2: []}
    for obj in range(n_objects):
        min_distance = float('inf')
        best_cluster = -1
        for k, prototype in prototypes.items():
            distance = sum(
                lambdas[k-1][j] * D[obj, prototype]
                for j, D in enumerate(D_matrices)
            )
            if distance < min_distance:
                min_distance = distance
                best_cluster = k
        new_clusters[best_cluster].append(obj)
    return new_clusters

# Algoritmo MRDCA-RWL
def mrdca_rwl(D_matrices, clusters, prototypes, max_iter=100, tol=1e-4):
    for iteration in range(max_iter):
        print(f"\nIteração {iteration + 1}:")
        
        # Atualizar pesos lambda
        lambdas = {}
        for k, cluster in clusters.items():
            lambdas[k] = calculate_lambda(cluster, prototypes[k], D_matrices)
            print(f"Lambdas para Cluster {k}: {lambdas[k]}")

        # Atualizar clusters
        new_clusters = update_clusters(D_matrices, lambdas, prototypes)

        # Atualizar protótipos
        for k, cluster in new_clusters.items():
            prototypes[k] = cluster[0]  # Simplesmente escolhemos o primeiro elemento
        
        # Verificar convergência
        if new_clusters == clusters:
            print("Convergiu!")
            break
        clusters = new_clusters

    return clusters, lambdas

# Rodar o algoritmo
final_clusters, final_lambdas = mrdca_rwl(D_matrices, clusters, prototypes)
print("\nClusters finais:", final_clusters)
print("Lambdas finais:", final_lambdas)