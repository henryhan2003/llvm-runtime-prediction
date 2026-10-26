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

struct io_snapshot {
    unsigned long long values[7];
    const char *status;
    char error[160];
};

static void read_child_io(pid_t child, struct io_snapshot *snapshot) {
    snapshot->status = "not_supported_platform";
#ifdef __linux__
    const char *names[] = {"rchar", "wchar", "syscr", "syscw", "read_bytes",
                           "write_bytes", "cancelled_write_bytes"};
    char path[64];
    snprintf(path, sizeof(path), "/proc/%ld/io", (long)child);
    FILE *stream = fopen(path, "r");
    if (stream == NULL) {
        int code = errno;
        snapshot->status = (code == EACCES || code == EPERM) ? "permission_denied" : "unavailable_proc_io";
        snprintf(snapshot->error, sizeof(snapshot->error), "%s", strerror(code));
        return;
    }
    unsigned int found = 0;
    char line[256], name[64];
    unsigned long long value;
    while (fgets(line, sizeof(line), stream) != NULL) {
        if (sscanf(line, "%63[^:]: %llu", name, &value) != 2) {
            continue;
        }
        for (unsigned int i = 0; i < 7; i++) {
            if (strcmp(name, names[i]) == 0) {
                snapshot->values[i] = value;
                found |= 1U << i;
            }
        }
    }
    int read_failed = ferror(stream);
    fclose(stream);
    snapshot->status = !read_failed && found == 127 ? "ok" : "invalid_proc_io";
    if (strcmp(snapshot->status, "ok") != 0) {
        snprintf(snapshot->error, sizeof(snapshot->error), "incomplete or unreadable /proc/PID/io");
    }
#else
    (void)child;
#endif
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

    struct io_snapshot io = {{0}, "not_collected", ""};
#ifdef __linux__
    /* Retain the exited child so even sub-millisecond commands have final I/O counters. */
    siginfo_t info;
    int wait_result;
    do {
        wait_result = waitid(P_PID, (id_t)child, &info, WEXITED | WNOWAIT);
    } while (wait_result < 0 && errno == EINTR);
    if (wait_result < 0) {
        perror("waitid");
        kill(child, SIGKILL);
        while (waitpid(child, NULL, 0) < 0 && errno == EINTR) {}
        return 125;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &finished) != 0) {
        perror("clock_gettime");
        while (waitpid(child, NULL, 0) < 0 && errno == EINTR) {}
        return 125;
    }
    read_child_io(child, &io);
#endif
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
#ifndef __linux__
    if (clock_gettime(CLOCK_MONOTONIC, &finished) != 0) {
        perror("clock_gettime");
        return 125;
    }
    read_child_io(child, &io);
#endif

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
        "probe_version=2\n"
        "child_pid=%ld\n"
        "elapsed_ns=%lld\n"
        "returncode=%d\n"
        "user_us=%lld\n"
        "system_us=%lld\n"
        "max_rss_bytes=%lld\n"
        "major_faults=%ld\n"
        "minor_faults=%ld\n"
        "fs_inputs=%ld\n"
        "fs_outputs=%ld\n"
        "io_status=%s\n"
        "io_error=%s\n"
        "io_rchar=%llu\n"
        "io_wchar=%llu\n"
        "io_syscr=%llu\n"
        "io_syscw=%llu\n"
        "io_read_bytes=%llu\n"
        "io_write_bytes=%llu\n"
        "io_cancelled_write_bytes=%llu\n",
        (long)child,
        timespec_diff_ns(&started, &finished),
        returncode,
        timeval_to_us(&usage.ru_utime),
        timeval_to_us(&usage.ru_stime),
        max_rss_bytes,
        usage.ru_majflt,
        usage.ru_minflt,
        usage.ru_inblock,
        usage.ru_oublock,
        io.status,
        io.error,
        io.values[0], io.values[1], io.values[2], io.values[3],
        io.values[4], io.values[5], io.values[6]
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
