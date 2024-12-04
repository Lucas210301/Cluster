#include <iostream>
#include <vector>
#include <algorithm> // Necessário para std::max_element
#include <iterator>  // Necessário para std::distance
#include <limits>    // Necessário para std::numeric_limits

// Função para realizar o clustering multi-visão com k-means
void multiViewKMeans(const std::vector<std::vector<double>>& view1,
                     const std::vector<std::vector<double>>& view2,
                     int k) {
    // Supondo que view1 e view2 sejam as duas visões dos dados, ambos são vetores 2D
    // Exemplo simples de inicialização para clusters:
    int n = view1.size(); // Número de dados

    std::vector<int> clusters(n, 0); // Atribuindo todos inicialmente ao cluster 0

    // Definir os protótipos como os primeiros pontos para simplificação
    std::vector<std::vector<double>> prototypes(k, std::vector<double>(view1[0].size(), 0));

    // Atribuição inicial aleatória de protótipos (simplesmente como os primeiros pontos):
    prototypes[0] = view1[0];
    prototypes[1] = view2[0];

    // Algoritmo de k-means simples para cada visão
    for (int iter = 0; iter < 10; ++iter) {
        // Passo 1: Atribuição de clusters
        for (int i = 0; i < n; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;

            // Encontrar o cluster mais próximo baseado na primeira visão (view1)
            for (int j = 0; j < k; ++j) {
                double dist = 0.0;
                for (size_t m = 0; m < view1[i].size(); ++m) {
                    dist += (view1[i][m] - prototypes[j][m]) * (view1[i][m] - prototypes[j][m]);
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            clusters[i] = best_cluster;
        }

        // Passo 2: Atualização dos protótipos
        for (int j = 0; j < k; ++j) {
            std::vector<double> new_prototype(view1[0].size(), 0.0);
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (clusters[i] == j) {
                    for (size_t m = 0; m < view1[i].size(); ++m) {
                        new_prototype[m] += view1[i][m];
                    }
                    count++;
                }
            }

            // Calcula o novo protótipo
            if (count > 0) {
                for (size_t m = 0; m < new_prototype.size(); ++m) {
                    prototypes[j][m] = new_prototype[m] / count;
                }
            }
        }

        // Passo 3: Exibindo os protótipos e clusters
        std::cout << "Iteração " << iter + 1 << ":\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "Ponto " << i + 1 << " pertence ao cluster " << clusters[i] + 1 << "\n";
        }
    }

    // Encontre o maior protótipo em termos de distância
    auto maxElement = std::max_element(prototypes.begin(), prototypes.end(),
                                       [](const std::vector<double>& a, const std::vector<double>& b) {
                                           double sum_a = 0, sum_b = 0;
                                           for (size_t i = 0; i < a.size(); ++i) {
                                               sum_a += a[i] * a[i];
                                               sum_b += b[i] * b[i];
                                           }
                                           return sum_a < sum_b;
                                       });

    int maxIndex = std::distance(prototypes.begin(), maxElement);
    std::cout << "O protótipo com maior distância é o protótipo " << maxIndex + 1 << "\n";
}

int main() {
    // Exemplo de duas visões de dados
    std::vector<std::vector<double>> view1 = {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0},
        {1.0, 0.6}
    };

    std::vector<std::vector<double>> view2 = {
        {0.0, 1.0},
        {0.1, 0.9},
        {1.0, 0.0},
        {0.9, 0.1},
        {1.5, 2.0}
    };

    int k = 2; // Número de clusters

    // Chamada da função de k-means multi-visão
    multiViewKMeans(view1, view2, k);

    return 0;
}
