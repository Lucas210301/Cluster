import numpy as np
import os
import random
from datetime import datetime

# Verificar e instalar dependências necessárias
def install_required_packages():
    """Instala pacotes necessários se não estiverem disponíveis"""
    import subprocess
    import sys
    
    required_packages = ['pandas', 'scikit-learn', 'openpyxl']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Instalar dependências
try:
    install_required_packages()
    import pandas as pd
    from sklearn.datasets import load_digits, load_iris, load_wine
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
except ImportError as e:
    print(f"Erro ao importar bibliotecas: {e}")
    print("Execute: pip install pandas scikit-learn openpyxl")
    exit(1)

# Criar diretório de saída se não existir
if not os.path.exists('resultados_mrdca'):
    os.makedirs('resultados_mrdca')

# Função para calcular os pesos de relevância lambda
def calculate_lambda(cluster, prototype, d_matrices):
    """
    Calcula os pesos lambda para um cluster
    
    Args:
        cluster: Lista de objetos do cluster
        prototype: Índice do protótipo
        d_matrices: Lista de matrizes de dissimilaridade
    
    Returns:
        Lista de pesos lambda normalizados
    """
    p = len(d_matrices)
    total_d = [sum(d_matrix[obj, prototype] for obj in cluster) for d_matrix in d_matrices]
    product = np.prod(total_d)
    
    # Evitar divisão por zero
    if product == 0:
        lambdas = [1.0 / p] * p
    else:
        lambdas = [(product ** (1 / p)) / d if d != 0 else 1e-10 for d in total_d]
        # Normalizar os pesos para que a soma seja 1
        total_lambda = sum(lambdas)
        if total_lambda == 0:
            lambdas = [1.0 / p] * p
        else:
            lambdas = [l / total_lambda for l in lambdas]
    
    return lambdas

def update_clusters(d_matrices, lambdas, prototypes):
    """
    Atualiza os clusters baseado nas distâncias ponderadas
    
    Args:
        d_matrices: Lista de matrizes de dissimilaridade
        lambdas: Dicionário de pesos lambda por cluster
        prototypes: Dicionário de protótipos por cluster
    
    Returns:
        Dicionário com novos clusters
    """
    n_objects = d_matrices[0].shape[0]
    new_clusters = {k: [] for k in prototypes.keys()}
    
    for obj in range(n_objects):
        min_distance = float('inf')
        best_cluster = 1  # Cluster padrão
        
        for k, prototype in prototypes.items():
            distance = sum(lambdas[k][j] * d_matrices[j][obj, prototype] for j in range(len(d_matrices)))
            if distance < min_distance:
                min_distance = distance
                best_cluster = k
        
        new_clusters[best_cluster].append(obj)
    
    return new_clusters

def select_prototypes(clusters, d_matrices):
    """
    Seleciona protótipos que minimizam a distância intra-cluster
    
    Args:
        clusters: Dicionário de clusters
        d_matrices: Lista de matrizes de dissimilaridade
    
    Returns:
        Dicionário de protótipos por cluster
    """
    prototypes = {}
    for k, cluster in clusters.items():
        if not cluster:
            prototypes[k] = 0  # Protótipo padrão se cluster vazio
            continue
            
        # Selecionar o objeto que minimiza a soma das distâncias para todos os outros objetos do cluster
        min_total_distance = float('inf')
        best_prototype = cluster[0]
        
        for candidate in cluster:
            total_distance = 0
            for obj in cluster:
                for d_matrix in d_matrices:
                    total_distance += d_matrix[candidate, obj]
            
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_prototype = candidate
        
        prototypes[k] = best_prototype
    
    return prototypes

def mrdca_rwl(d_matrices, n_clusters, max_iter=30, verbose=False, random_state=None):
    """
    Algoritmo MRDCA-RWL principal
    
    Args:
        d_matrices: Lista de matrizes de dissimilaridade
        n_clusters: Número de clusters
        max_iter: Número máximo de iterações (fixado em 30)
        verbose: Se True, imprime informações detalhadas
        random_state: Seed para reprodutibilidade
    
    Returns:
        Tupla (clusters, lambdas, iterations)
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    n_objects = d_matrices[0].shape[0]
    
    # Inicialização aleatória
    clusters = {k: [] for k in range(1, n_clusters + 1)}
    objects = list(range(n_objects))
    random.shuffle(objects)
    
    # Distribuir objetos aleatoriamente entre clusters
    for i, obj in enumerate(objects):
        cluster_id = (i % n_clusters) + 1
        clusters[cluster_id].append(obj)
    
    # Selecionar protótipos iniciais
    prototypes = select_prototypes(clusters, d_matrices)
    
    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        if verbose:
            print(f"\nIteração {iteration + 1}:")
        
        # Atualizar pesos lambda
        lambdas = {}
        for k, cluster in clusters.items():
            if cluster:  # Verificar se o cluster não está vazio
                lambdas[k] = calculate_lambda(cluster, prototypes[k], d_matrices)
                if verbose:
                    print(f"Lambdas para Cluster {k}: {lambdas[k]}")
            else:
                lambdas[k] = [1.0 / len(d_matrices)] * len(d_matrices)
        
        # Atualizar clusters
        new_clusters = update_clusters(d_matrices, lambdas, prototypes)
        
        # Atualizar protótipos
        new_prototypes = select_prototypes(new_clusters, d_matrices)
        
        # Verificar convergência
        if new_clusters == clusters:
            if verbose:
                print("Convergiu!")
            break
        
        clusters = new_clusters
        prototypes = new_prototypes
    
    return clusters, lambdas, iterations

def calculate_metrics(clusters, true_labels, data):
    """
    Calcula métricas de avaliação dos clusters
    
    Args:
        clusters: Dicionário de clusters
        true_labels: Labels verdadeiros
        data: Dados originais
    
    Returns:
        Tupla (ari, nmi, silhouette)
    """
    # Converter clusters para array de labels
    predicted_labels = np.zeros(len(true_labels))
    for cluster_id, objects in clusters.items():
        for obj in objects:
            predicted_labels[obj] = cluster_id - 1
    
    # Calcular métricas
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    try:
        silhouette = silhouette_score(data, predicted_labels)
    except ValueError:
        silhouette = -1  # Valor padrão se não conseguir calcular
    
    return ari, nmi, silhouette

def create_dissimilarity_matrices(data):
    """
    Cria matrizes de dissimilaridade usando diferentes métricas
    
    Args:
        data: Dados de entrada
    
    Returns:
        Lista de matrizes de dissimilaridade
    """
    d1 = pairwise_distances(data, metric='euclidean')
    d2 = pairwise_distances(data, metric='cityblock')
    return [d1, d2]

def run_single_test(data, true_labels, n_clusters, test_number, random_state=None):
    """
    Executa um único teste do algoritmo
    
    Args:
        data: Dados de entrada
        true_labels: Labels verdadeiros
        n_clusters: Número de clusters
        test_number: Número do teste
        random_state: Seed para reprodutibilidade
    
    Returns:
        Dicionário com resultados do teste
    """
    # Criar matrizes de dissimilaridade
    d_matrices = create_dissimilarity_matrices(data)
    
    # Executar algoritmo
    clusters, lambdas, iterations = mrdca_rwl(d_matrices, n_clusters, 
                                             max_iter=30, verbose=False, 
                                             random_state=random_state)
    
    # Calcular métricas
    ari, nmi, silhouette = calculate_metrics(clusters, true_labels, data)
    
    # Calcular média dos lambdas
    lambda_means = []
    if lambdas:
        for i in range(len(d_matrices)):
            lambda_sum = sum(lambdas[k][i] for k in lambdas.keys() if k in lambdas and lambdas[k])
            lambda_means.append(lambda_sum / len(lambdas) if lambdas else 0)
    else:
        lambda_means = [0.0, 0.0]
    
    return {
        'Teste': test_number,
        'ARI': ari,
        'NMI': nmi,
        'Silhouette': silhouette,
        'Lambda1_Media': lambda_means[0] if len(lambda_means) > 0 else 0,
        'Lambda2_Media': lambda_means[1] if len(lambda_means) > 1 else 0,
        'N_Iteracoes': iterations
    }

def run_multiple_tests(data, true_labels, n_clusters, dataset_name, n_tests=50):
    """
    Executa múltiplos testes para uma base de dados
    
    Args:
        data: Dados de entrada
        true_labels: Labels verdadeiros
        n_clusters: Número de clusters
        dataset_name: Nome da base de dados
        n_tests: Número de testes a executar
    
    Returns:
        Lista de resultados
    """
    print(f"\nExecutando {n_tests} testes para {dataset_name}...")
    
    results = []
    
    for test in range(n_tests):
        print(f"Teste {test + 1}/{n_tests} - {dataset_name}")
        
        # Usar seed diferente para cada teste
        result = run_single_test(data, true_labels, n_clusters, test + 1, 
                               random_state=42 + test)
        results.append(result)
    
    return results

def calculate_statistics(results):
    """
    Calcula estatísticas resumo dos resultados
    
    Args:
        results: Lista de resultados dos testes
    
    Returns:
        DataFrame com estatísticas
    """
    df = pd.DataFrame(results)
    
    stats = {
        'Metrica': ['ARI', 'NMI', 'Silhouette', 'Lambda1_Media', 'Lambda2_Media', 'N_Iteracoes'],
        'Media': [
            df['ARI'].mean(),
            df['NMI'].mean(),
            df['Silhouette'].mean(),
            df['Lambda1_Media'].mean(),
            df['Lambda2_Media'].mean(),
            df['N_Iteracoes'].mean()
        ],
        'Desvio_Padrao': [
            df['ARI'].std(),
            df['NMI'].std(),
            df['Silhouette'].std(),
            df['Lambda1_Media'].std(),
            df['Lambda2_Media'].std(),
            df['N_Iteracoes'].std()
        ],
        'Minimo': [
            df['ARI'].min(),
            df['NMI'].min(),
            df['Silhouette'].min(),
            df['Lambda1_Media'].min(),
            df['Lambda2_Media'].min(),
            df['N_Iteracoes'].min()
        ],
        'Maximo': [
            df['ARI'].max(),
            df['NMI'].max(),
            df['Silhouette'].max(),
            df['Lambda1_Media'].max(),
            df['Lambda2_Media'].max(),
            df['N_Iteracoes'].max()
        ]
    }
    
    return pd.DataFrame(stats)

def save_results(results, stats, dataset_name, timestamp):
    """
    Salva os resultados em arquivos
    
    Args:
        results: Lista de resultados
        stats: DataFrame com estatísticas
        dataset_name: Nome da base de dados
        timestamp: Timestamp para nomes únicos
    """
    df_results = pd.DataFrame(results)
    
    # Salvar resultados detalhados em Excel
    excel_filename = f'resultados_mrdca/resultados_detalhados_{dataset_name}_{timestamp}.xlsx'
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Resultados_Detalhados', index=False)
            stats.to_excel(writer, sheet_name='Estatisticas', index=False)
        print(f"Arquivo Excel salvo: {excel_filename}")
    except Exception as e:
        print(f"Erro ao salvar Excel: {e}")
    
    # Salvar em TXT
    txt_filename = f'resultados_mrdca/resultados_{dataset_name}_{timestamp}.txt'
    try:
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"RESULTADOS PARA BASE DE DADOS: {dataset_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write("ESTATÍSTICAS RESUMO:\n")
            f.write("-"*30 + "\n")
            f.write(stats.to_string(index=False))
            f.write("\n\n")
            
            f.write("RESULTADOS DETALHADOS:\n")
            f.write("-"*30 + "\n")
            f.write(df_results.to_string(index=False))
        print(f"Arquivo TXT salvo: {txt_filename}")
    except Exception as e:
        print(f"Erro ao salvar TXT: {e}")

def main():
    """Função principal do programa"""
    # Fixar seed para reprodutibilidade
    np.random.seed(42)
    random.seed(42)
    
    # Carregando as bases de dados
    print("Carregando bases de dados...")
    try:
        iris = load_iris()
        wine = load_wine()
        digits = load_digits()
    except Exception as e:
        print(f"Erro ao carregar bases de dados: {e}")
        return
    
    datasets = [
        (iris.data, iris.target, 3, 'Iris'),
        (wine.data, wine.target, 3, 'Wine'),
        (digits.data[:500], digits.target[:500], 10, 'Digits')  # Usando apenas 500 amostras do digits
    ]
    
    # Timestamp para nomes de arquivos únicos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Arquivo de resumo executivo
    resumo_filename = f'resultados_mrdca/resumo_executivos_{timestamp}.txt'
    try:
        with open(resumo_filename, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO EXECUTIVO - ALGORITMO MRDCA-RWL\n")
            f.write("="*50 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write("Número de testes por base: 50\n")
            f.write("Critério de parada: 30 iterações ou convergência\n\n")
    except Exception as e:
        print(f"Erro ao criar arquivo de resumo: {e}")
        return
    
    for data, labels, n_clusters, name in datasets:
        print(f"\n{'='*50}")
        print(f"PROCESSANDO BASE: {name}")
        print(f"{'='*50}")
        
        try:
            # Executar testes
            results = run_multiple_tests(data, labels, n_clusters, name, 50)
            
            # Calcular estatísticas
            stats = calculate_statistics(results)
            
            # Salvar resultados
            save_results(results, stats, name, timestamp)
            
            # Adicionar ao resumo executivo
            with open(resumo_filename, 'a', encoding='utf-8') as f:
                f.write(f"\nBASE DE DADOS: {name}\n")
                f.write("-"*30 + "\n")
                f.write(f"Número de objetos: {len(data)}\n")
                f.write(f"Número de clusters: {n_clusters}\n")
                f.write(f"ARI Médio: {stats[stats['Metrica'] == 'ARI']['Media'].iloc[0]:.4f} ± {stats[stats['Metrica'] == 'ARI']['Desvio_Padrao'].iloc[0]:.4f}\n")
                f.write(f"NMI Médio: {stats[stats['Metrica'] == 'NMI']['Media'].iloc[0]:.4f} ± {stats[stats['Metrica'] == 'NMI']['Desvio_Padrao'].iloc[0]:.4f}\n")
                f.write(f"Silhouette Médio: {stats[stats['Metrica'] == 'Silhouette']['Media'].iloc[0]:.4f} ± {stats[stats['Metrica'] == 'Silhouette']['Desvio_Padrao'].iloc[0]:.4f}\n")
                f.write(f"Iterações Médias: {stats[stats['Metrica'] == 'N_Iteracoes']['Media'].iloc[0]:.2f} ± {stats[stats['Metrica'] == 'N_Iteracoes']['Desvio_Padrao'].iloc[0]:.2f}\n")
                f.write("\n")
            
        except Exception as e:
            print(f"Erro ao processar base {name}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("EXECUÇÃO CONCLUÍDA!")
    print(f"{'='*50}")
    print(f"Resumo executivo salvo em: {resumo_filename}")
    print("Todos os arquivos foram salvos na pasta 'resultados_mrdca/'")

if __name__ == "__main__":
    main()