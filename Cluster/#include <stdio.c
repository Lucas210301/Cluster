#include <stdio.h>
#include <math.h>
#include <float.h>

#define N 4 // Número de objetos
#define P 2 // Número de matrizes de dissimilaridade
#define K 2 // Número de clusters

// Matrizes de dissimilaridade
double D1[N][N] = {
    {0, 1, 2, 3},
    {1, 0, 2, 2},
    {2, 2, 0, 1},
    {3, 2, 1, 0}
};
double D2[N][N] = {
    {0, 2, 1, 4},
    {2, 0, 3, 3},
    {1, 3, 0, 2},
    {4, 3, 2, 0}
};
double* D_matrices[] = {D1, D2};

// Inicialização
int clusters[K][N] = {{0, 1}, {2, 3}};
int prototypes[K] = {0, 3};

void calculate_lambda(int cluster[], int cluster_size, int prototype, double* lambdas) {
    double total_d[P] = {0.0};
    double product = 1.0;

    // Soma das dissimilaridades
    for (int j = 0; j < P; j++) {
        for (int i = 0; i < cluster_size; i++) {
            total_d[j] += D_matrices[j][cluster[i]][prototype];
        }
        product *= total_d[j];
    }

    // Calcular lambdas
    for (int j = 0; j < P; j++) {
        lambdas[j] = pow(product, 1.0 / P) / total_d[j];
    }
}

int main() {
    // Exemplo expandido
    double lambdas[K][P];
    for (int k = 0; k < K; k++) {
        calculate_lambda(clusters[k], 2, prototypes[k], lambdas[k]);
        printf("Lambdas para Cluster %d: %.3lf, %.3lf\n", k + 1, lambdas[k][0], lambdas[k][1]);
    }
    return 0;
}
