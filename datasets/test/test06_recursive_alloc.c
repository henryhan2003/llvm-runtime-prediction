#include <stdio.h>
#include <stdlib.h>

#define N 24

static int fib_like(int n) {
    if (n <= 1) {
        return n;
    }
    return fib_like(n - 1) + fib_like(n - 2);
}

static int allocated_sum(void) {
    int *values = (int *)malloc(sizeof(int) * N);
    int total = 0;

    if (values == NULL) {
        return -1;
    }

    for (int i = 0; i < N; ++i) {
        values[i] = fib_like(i % 12);
        total += values[i];
    }

    free(values);
    return total;
}

int main(void) {
    printf("%d\n", allocated_sum());
    return 0;
}
