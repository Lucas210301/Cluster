import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.metrics.pairwise import pairwise_distances

# Constantes para literais duplicados
CLUSTERS_LITERAL = "Clusters:"
LAMBDAS_LITERAL = "Lambdas:"

# Função para calcular os pesos de relevância lambda
def calculate_lambda(cluster, prototype, d_matrices):
    p = len(d_matrices)  # Número de matrizes de dissimilaridade
    total_d = [sum(d_matrix[obj, prototype] for obj in cluster) for d_matrix in d_matrices]
    product = np.prod(total_d)
    
    # Evitar divisão por zero
    if product == 0:
        lambdas = [1.0 / p] * p  # Distribui igualmente os pesos se o produto for zero
    else:
        lambdas = [(product ** (1 / p)) / d if d != 0 else 0 for d in total_d]
        # Normalizar os pesos para que a soma seja 1
        total_lambda = sum(lambdas)
        if total_lambda == 0:
            lambdas = [1.0 / p] * p
        else:
            lambdas = [l / total_lambda for l in lambdas]
    
    return lambdas

# Função para atualizar os clusters
def update_clusters(d_matrices, lambdas, prototypes):
    n_objects = d_matrices[0].shape[0]
    new_clusters = {k: [] for k in prototypes.keys()}
    
    for obj in range(n_objects):
        min_distance = float('inf')
        best_cluster = -1
        
        for k, prototype in prototypes.items():
            distance = sum(lambdas[k][j] * d_matrices[j][obj, prototype] for j in range(len(d_matrices)))
            if distance < min_distance:
                min_distance = distance
                best_cluster = k
        
        if best_cluster == -1:
            raise ValueError("Nenhum cluster adequado encontrado para o objeto.")
        
        new_clusters[best_cluster].append(obj)
    
    return new_clusters

# Função principal do algoritmo MRDCA-RWL
def mrdca_rwl(d_matrices, clusters, prototypes, max_iter=100):
    for iteration in range(max_iter):
        print(f"\nIteração {iteration + 1}:")
        
        # Atualizar pesos lambda
        lambdas = {}
        for k, cluster in clusters.items():
            lambdas[k] = calculate_lambda(cluster, prototypes[k], d_matrices)
            print(f"{LAMBDAS_LITERAL} para Cluster {k}: {lambdas[k]}")
        
        # Atualizar clusters
        new_clusters = update_clusters(d_matrices, lambdas, prototypes)
        
        # Atualizar protótipos
        for k, cluster in new_clusters.items():
            if cluster:  # Verifica se o cluster não está vazio
                prototypes[k] = cluster[0]  # Escolhe o primeiro elemento como protótipo
        
        # Verificar convergência
        if new_clusters == clusters:
            print("Convergiu!")
            break
        
        clusters = new_clusters
    
    return clusters, lambdas

# Função para criar matrizes de dissimilaridade
def create_dissimilarity_matrices(data):
    d1 = pairwise_distances(data, metric='euclidean')
    d2 = pairwise_distances(data, metric='cityblock')
    return [d1, d2]

# Função para inicializar clusters e protótipos
def initialize_clusters(n_clusters):
    clusters = {i: [] for i in range(1, n_clusters + 1)}
    prototypes = {i: i for i in range(1, n_clusters + 1)}
    return clusters, prototypes

# Função para rodar o algoritmo MRDCA-RWL
def run_mrdca_rwl(d_matrices, n_clusters):
    n_objects = d_matrices[0].shape[0]
    clusters, prototypes = initialize_clusters(n_clusters)
    final_clusters, final_lambdas = mrdca_rwl(d_matrices, clusters, prototypes)
    return final_clusters, final_lambdas

# Carregando as bases de dados
iris = load_iris()
wine = load_wine()
digits = load_digits()

# Criando matrizes de dissimilaridade para cada base de dados
iris_matrices = create_dissimilarity_matrices(iris.data)
wine_matrices = create_dissimilarity_matrices(wine.data)
digits_matrices = create_dissimilarity_matrices(digits.data)

# Executando o algoritmo para as três bases de dados
iris_clusters, iris_lambdas = run_mrdca_rwl(iris_matrices, n_clusters=3)
wine_clusters, wine_lambdas = run_mrdca_rwl(wine_matrices, n_clusters=3)
digits_clusters, digits_lambdas = run_mrdca_rwl(digits_matrices, n_clusters=10)

# Exibindo os resultados
print("\nResultados para Iris:")
print(f"{CLUSTERS_LITERAL}", iris_clusters)
print(f"{LAMBDAS_LITERAL}", iris_lambdas)

print("\nResultados para Wine:")
print(f"{CLUSTERS_LITERAL}", wine_clusters)
print(f"{LAMBDAS_LITERAL}", wine_lambdas)

print("\nResultados para Digits:")
print(f"{CLUSTERS_LITERAL}", digits_clusters)
print(f"{LAMBDAS_LITERAL}", digits_lambdas)