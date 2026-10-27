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

static void read_child_io(const char *path, pid_t child, struct io_snapshot *snapshot) {
    const char *names[] = {"io_rchar", "io_wchar", "io_syscr", "io_syscw", "io_read_bytes",
                           "io_write_bytes", "io_cancelled_write_bytes"};
    FILE *stream = fopen(path, "r");
    if (stream == NULL) {
        snapshot->status = "missing_self_io_snapshot";
        snprintf(snapshot->error, sizeof(snapshot->error), "no normal-exit self I/O report (static binary, _exit, signal, or preload failure)");
        return;
    }
    unsigned int found = 0;
    long report_pid = -1;
    int collector_version = -1;
    char report_status[64] = "";
    char line[256], name[64];
    unsigned long long value;
    while (fgets(line, sizeof(line), stream) != NULL) {
        if (sscanf(line, "io_collector_version=%d", &collector_version) == 1) {
            continue;
        }
        if (sscanf(line, "io_child_pid=%ld", &report_pid) == 1) {
            continue;
        }
        if (sscanf(line, "io_status=%63s", report_status) == 1) {
            continue;
        }
        if (strncmp(line, "io_error=", 9) == 0) {
            snprintf(snapshot->error, sizeof(snapshot->error), "%.159s", line + 9);
            snapshot->error[strcspn(snapshot->error, "\r\n")] = '\0';
            continue;
        }
        if (sscanf(line, "%63[^=]=%llu", name, &value) != 2) {
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
    if (read_failed || report_pid != (long)child || report_status[0] == '\0') {
        snapshot->status = "invalid_self_io_snapshot";
        snprintf(snapshot->error, sizeof(snapshot->error), "incomplete or mismatched-PID self I/O report");
    } else if (collector_version != 4) {
        snapshot->status = "incompatible_self_io_collector";
        snprintf(snapshot->error, sizeof(snapshot->error), "expected v4 self I/O collector; rebuild probe and library together");
    } else if (strcmp(report_status, "ok") != 0) {
        snapshot->status = "self_io_collection_failed";
    } else {
        snapshot->status = found == 127 ? "ok" : "invalid_self_io_snapshot";
        if (found != 127) {
            snprintf(snapshot->error, sizeof(snapshot->error), "incomplete self I/O counters");
        }
    }
}

int main(int argc, char **argv) {
    int command_index = 3;
    const char *collector = NULL;
    if (argc >= 6 && strcmp(argv[2], "--io-collector") == 0 && strcmp(argv[4], "--") == 0) {
        collector = argv[3];
        command_index = 5;
    } else if (argc < 4 || strcmp(argv[2], "--") != 0) {
        fprintf(stderr, "usage: %s METRICS_FILE [--io-collector LIBRARY] -- COMMAND [ARG ...]\n", argv[0]);
        return 125;
    }
    size_t path_size = strlen(argv[1]) + 4;
    char *io_path = malloc(path_size);
    if (io_path == NULL) {
        return 125;
    }
    snprintf(io_path, path_size, "%s.io", argv[1]);
    unlink(io_path);

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
        if (collector != NULL) {
            char pid_text[32];
            snprintf(pid_text, sizeof(pid_text), "%ld", (long)getpid());
            const char *existing = getenv("LD_PRELOAD");
            size_t size = strlen(collector) + (existing != NULL ? strlen(existing) : 0) + 2;
            char *preload = malloc(size);
            if (preload == NULL) {
                _exit(125);
            }
            snprintf(preload, size, "%s%s%s", collector, existing != NULL && *existing ? ":" : "",
                     existing != NULL ? existing : "");
            if (setenv("LD_PRELOAD", preload, 1) != 0 || setenv("LLVM_RUNTIME_IO_REPORT", io_path, 1) != 0 ||
                setenv("LLVM_RUNTIME_IO_PID", pid_text, 1) != 0) {
                perror("setenv self I/O collector");
                _exit(125);
            }
            free(preload);
        }
        execvp(argv[command_index], &argv[command_index]);
        perror("execvp");
        _exit(127);
    }

    struct io_snapshot io = {{0}, "not_collected", ""};
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
    if (collector != NULL) {
        read_child_io(io_path, child, &io);
    } else {
        io.status = "not_collected_no_self_collector";
    }
    unlink(io_path);
    free(io_path);

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
        "probe_version=4\n"
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
