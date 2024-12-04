#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

// Função para calcular a distância Euclidiana
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// Calcular a média da distância intra-cluster
double meanIntraClusterDistance(const std::vector<std::vector<double>>& data, 
                                 const std::vector<int>& labels, int cluster, int pointIndex) {
    double totalDistance = 0.0;
    int count = 0;

    for (size_t i = 0; i < data.size(); ++i) {
        if (labels[i] == cluster && i != pointIndex) {
            totalDistance += euclideanDistance(data[pointIndex], data[i]);
            count++;
        }
    }

    return count > 0 ? totalDistance / count : 0.0;
}

// Calcular a distância média para o cluster mais próximo
double meanNearestClusterDistance(const std::vector<std::vector<double>>& data, 
                                   const std::vector<int>& labels, int pointIndex, int currentCluster) {
    double minDistance = std::numeric_limits<double>::max();

    for (size_t c = 0; c < data.size(); ++c) {
        if (labels[c] != currentCluster) {
            double distance = euclideanDistance(data[pointIndex], data[c]);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    }

    return minDistance;
}

// Calcular índice de Silhouette
double silhouetteScore(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    double totalScore = 0.0;
    int n = data.size();

    for (int i = 0; i < n; ++i) {
        int currentCluster = labels[i];

        // Coesão (a)
        double a = meanIntraClusterDistance(data, labels, currentCluster, i);

        // Separação (b)
        double b = meanNearestClusterDistance(data, labels, i, currentCluster);

        // Silhouette
        double s = (b - a) / std::max(a, b);
        totalScore += s;
    }

    return totalScore / n;
}

// Calcular índice de Davies-Bouldin
double daviesBouldinScore(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int k) {
    std::vector<double> clusterDispersion(k, 0.0);
    std::vector<std::vector<double>> centroids(k, std::vector<double>(data[0].size(), 0.0));
    std::vector<int> clusterSizes(k, 0);

    // Calcular centróides e dispersões
    for (size_t i = 0; i < data.size(); ++i) {
        int cluster = labels[i];
        clusterSizes[cluster]++;
        for (size_t j = 0; j < data[0].size(); ++j) {
            centroids[cluster][j] += data[i][j];
        }
    }

    for (int c = 0; c < k; ++c) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            centroids[c][j] /= clusterSizes[c];
        }
    }

    for (size_t i = 0; i < data.size(); ++i) {
        int cluster = labels[i];
        clusterDispersion[cluster] += euclideanDistance(data[i], centroids[cluster]);
    }

    for (int c = 0; c < k; ++c) {
        clusterDispersion[c] /= clusterSizes[c];
    }

    // Calcular índice Davies-Bouldin
    double totalScore = 0.0;

    for (int i = 0; i < k; ++i) {
        double maxRatio = 0.0;
        for (int j = 0; j < k; ++j) {
            if (i != j) {
                double distance = euclideanDistance(centroids[i], centroids[j]);
                double ratio = (clusterDispersion[i] + clusterDispersion[j]) / distance;
                maxRatio = std::max(maxRatio, ratio);
            }
        }
        totalScore += maxRatio;
    }

    return totalScore / k;
}

int main() {
    // Dados simulados
    std::vector<std::vector<double>> data = {
        {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0}, 
        {1.0, 0.6}, {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0}
    };

    // Rótulos de cluster gerados por um algoritmo de clustering (exemplo)
    std::vector<int> labels = {0, 0, 1, 1, 0, 1, 2, 2};

    int k = 3; // Número de clusters

    // Calcular índices de validação
    double silhouette = silhouetteScore(data, labels);
    double daviesBouldin = daviesBouldinScore(data, labels, k);

    std::cout << "Índice de Silhouette: " << silhouette << "\n";
    std::cout << "Índice de Davies-Bouldin: " << daviesBouldin << "\n";

    return 0;
}
