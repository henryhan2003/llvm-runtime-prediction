#include <stdio.h>

#define N 32

static int matrix_checksum(void) {
    int a[N][N];
    int b[N][N];
    int c[N][N];
    int checksum = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = i + j;
            b[i][j] = i - j;
            c[i][j] = 0;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
            checksum += c[i][j];
        }
    }

    return checksum;
}

int main(void) {
    printf("%d\n", matrix_checksum());
    return 0;
}
