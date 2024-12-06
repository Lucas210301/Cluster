#include <stdio.h>
#define N 4

void update_similarity(double S[N][N], int partitions[][2], int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                S[partitions[i][j]][partitions[i][k]] += 1.0;
            }
        }
    }
}

int main() {
    double S[N][N] = {0};
    int partitions[2][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};

    // Atualizar matriz de similaridade
    update_similarity(S, partitions, 2);

    // Normalizar matriz
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            S[i][j] /= 2.0;
        }
    }

    // Exibir matriz de similaridade
    printf("Matriz de Similaridade:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", S[i][j]);
        }
        printf("\n");
    }

    return 0;
}
