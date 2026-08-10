#define _GNU_SOURCE

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static long long timespec_diff_ns(const struct timespec *start, const struct timespec *end) {
    return (long long)(end->tv_sec - start->tv_sec) * 1000000000LL
        + (long long)(end->tv_nsec - start->tv_nsec);
}

static long long timeval_to_us(const struct timeval *value) {
    return (long long)value->tv_sec * 1000000LL + (long long)value->tv_usec;
}

static int child_returncode(int status) {
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return -WTERMSIG(status);
    }
    return 125;
}

int main(int argc, char **argv) {
    if (argc < 4 || strcmp(argv[2], "--") != 0) {
        fprintf(stderr, "usage: %s METRICS_FILE -- COMMAND [ARG ...]\n", argv[0]);
        return 125;
    }

    struct timespec started;
    struct timespec finished;
    if (clock_gettime(CLOCK_MONOTONIC, &started) != 0) {
        perror("clock_gettime");
        return 125;
    }

    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return 125;
    }
    if (child == 0) {
        execvp(argv[3], &argv[3]);
        perror("execvp");
        _exit(127);
    }

    int status = 0;
    struct rusage usage;
    pid_t waited;
    do {
        waited = wait4(child, &status, 0, &usage);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        perror("wait4");
        return 125;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &finished) != 0) {
        perror("clock_gettime");
        return 125;
    }

#ifdef __APPLE__
    long long max_rss_bytes = (long long)usage.ru_maxrss;
#else
    long long max_rss_bytes = (long long)usage.ru_maxrss * 1024LL;
#endif
    int returncode = child_returncode(status);
    FILE *metrics = fopen(argv[1], "w");
    if (metrics == NULL) {
        perror("fopen metrics");
        return 125;
    }
    int write_failed = fprintf(
        metrics,
        "probe_version=1\n"
        "elapsed_ns=%lld\n"
        "returncode=%d\n"
        "user_us=%lld\n"
        "system_us=%lld\n"
        "max_rss_bytes=%lld\n"
        "major_faults=%ld\n"
        "minor_faults=%ld\n"
        "fs_inputs=%ld\n"
        "fs_outputs=%ld\n",
        timespec_diff_ns(&started, &finished),
        returncode,
        timeval_to_us(&usage.ru_utime),
        timeval_to_us(&usage.ru_stime),
        max_rss_bytes,
        usage.ru_majflt,
        usage.ru_minflt,
        usage.ru_inblock,
        usage.ru_oublock
    ) < 0;
    if (fclose(metrics) != 0 || write_failed) {
        perror("write metrics");
        return 125;
    }

    if (returncode < 0) {
        return 128 + (-returncode);
    }
    return returncode;
}
