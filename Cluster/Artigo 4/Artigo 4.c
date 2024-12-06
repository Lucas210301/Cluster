#include <stdio.h>
#include <math.h> // Inclui fmax e outras funções matemáticas

double calculate_silhouette(double distances[][6], int labels[], int n, int k) {
    double silhouette = 0.0;
    for (int i = 0; i < n; i++) {
        double a = 0.0, b = INFINITY;
        int cluster = labels[i];

        // Calcular a(i): Distância média para o mesmo cluster
        for (int j = 0; j < n; j++) {
            if (labels[j] == cluster && i != j) {
                a += distances[i][j];
            }
        }
        a /= n - 1;

        // Calcular b(i): Menor distância média para outro cluster
        for (int c = 0; c < k; c++) {
            if (c != cluster) {
                double dist_sum = 0.0, count = 0;
                for (int j = 0; j < n; j++) {
                    if (labels[j] == c) {
                        dist_sum += distances[i][j];
                        count++;
                    }
                }
                double avg_dist = dist_sum / count;
                if (avg_dist < b) b = avg_dist;
            }
        }

        // Índice de Silhueta para o ponto i
        silhouette += (b - a) / fmax(a, b); // Usa fmax para evitar divisão por zero
    }
    return silhouette / n; // Retorna a média de Silhueta
}

int main() {
    int labels[6] = {0, 0, 0, 1, 1, 1}; // Clusters atribuídos
    double distances[6][6] = {         // Matriz de distâncias entre pontos
        {0, 1, 1.4, 12.7, 14.1, 13.4},
        {1, 0, 1.4, 12.7, 14.1, 13.4},
        {1.4, 1.4, 0, 12.7, 14.1, 13.4},
        {12.7, 12.7, 12.7, 0, 1, 1.4},
        {14.1, 14.1, 14.1, 1, 0, 1.4},
        {13.4, 13.4, 13.4, 1.4, 1.4, 0},
    };

    double silhouette = calculate_silhouette(distances, labels, 6, 2);
    printf("Índice de Silhueta: %.2f\n", silhouette);

    return 0;
}
