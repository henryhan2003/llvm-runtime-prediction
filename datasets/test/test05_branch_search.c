#include <stdio.h>

#define N 256

static int branchy_search(int target) {
    int data[N];
    int score = 0;

    for (int i = 0; i < N; ++i) {
        data[i] = (i * 17 + 11) % 251;
    }

    for (int i = 0; i < N; ++i) {
        if (data[i] == target) {
            score += i;
        } else if (data[i] < target) {
            score -= data[i] & 7;
        } else {
            score += data[i] & 3;
        }
    }

    return score;
}

int main(void) {
    int total = 0;
    for (int t = 0; t < 64; ++t) {
        total += branchy_search(t);
    }
    printf("%d\n", total);
    return 0;
}
