#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

// Representação de matrizes de dissimilaridade
std::vector<std::vector<std::vector<double>>> createDissimilarityMatrices() {
    std::vector<std::vector<double>> D1 = {
        {0, 1, 2, 3},
        {1, 0, 2, 2},
        {2, 2, 0, 1},
        {3, 2, 1, 0}
    };

    std::vector<std::vector<double>> D2 = {
        {0, 2, 1, 4},
        {2, 0, 3, 3},
        {1, 3, 0, 2},
        {4, 3, 2, 0}
    };

    return {D1, D2};
}

std::vector<double> calculateLambda(const std::vector<int>& cluster, int prototype,
                                    const std::vector<std::vector<std::vector<double>>>& D_matrices) {
    int p = D_matrices.size(); // Número de matrizes
    std::vector<double> lambdas(p, 0.0);
    std::vector<double> total_d(p, 0.0);

    // Soma das dissimilaridades para cada matriz
    for (int j = 0; j < p; ++j) {
        for (int obj : cluster) {
            total_d[j] += D_matrices[j][obj][prototype];
        }
    }

    // Produto das somas
    double product = 1.0;
    for (double d : total_d) {
        product *= d;
    }

    // Calcular λ para cada matriz
    for (int j = 0; j < p; ++j) {
        lambdas[j] = pow(product, 1.0 / p) / total_d[j];
    }

    return lambdas;
}

int main() {
    auto D_matrices = createDissimilarityMatrices();

    // Cluster 1 com protótipo no índice 0
    std::vector<int> cluster1 = {0, 1};
    int prototype1 = 0;

    // Cluster 2 com protótipo no índice 3
    std::vector<int> cluster2 = {2, 3};
    int prototype2 = 3;

    // Calcular λ para cada cluster
    auto lambda1 = calculateLambda(cluster1, prototype1, D_matrices);
    auto lambda2 = calculateLambda(cluster2, prototype2, D_matrices);

    std::cout << "Lambdas para Cluster 1:\n";
    for (double lambda : lambda1) {
        std::cout << lambda << " ";
    }
    std::cout << "\n";

    std::cout << "Lambdas para Cluster 2:\n";
    for (double lambda : lambda2) {
        std::cout << lambda << " ";
    }
    std::cout << "\n";

    return 0;
}