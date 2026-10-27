#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

static char *report_path;
static pid_t target_pid;

__attribute__((constructor))
static void configure_collector(void) {
    const char *path = getenv("LLVM_RUNTIME_IO_REPORT");
    const char *pid = getenv("LLVM_RUNTIME_IO_PID");
    if (path != NULL && pid != NULL && (pid_t)strtol(pid, NULL, 10) == getpid()) {
        report_path = strdup(path);
        target_pid = getpid();
    }
}

__attribute__((destructor))
static void collect_self_io(void) {
    if (report_path == NULL || target_pid != getpid()) {
        return;
    }
    int saved_errno = errno;
    /* Include pending C stdio output, but snapshot before writing our own report. */
    fflush(NULL);
    const char *status = "ok";
    char error[160] = "";
    const char *names[] = {"rchar", "wchar", "syscr", "syscw", "read_bytes",
                           "write_bytes", "cancelled_write_bytes"};
    unsigned long long values[7] = {0};
    unsigned int found = 0;
    char buffer[2048];
    /* Benchmarks can export non-POSIX read/write; bypass their ELF symbols. */
    int input = (int)syscall(SYS_openat, AT_FDCWD, "/proc/self/io", O_RDONLY | O_CLOEXEC, 0);
    if (input < 0) {
        status = (errno == EACCES || errno == EPERM) ? "permission_denied" : "unavailable_proc_io";
        snprintf(error, sizeof(error), "%s", strerror(errno));
    } else {
        ssize_t count;
        do {
            count = (ssize_t)syscall(SYS_read, input, buffer, sizeof(buffer) - 1);
        } while (count < 0 && errno == EINTR);
        if (count <= 0) {
            status = "invalid_proc_io";
            snprintf(error, sizeof(error), "cannot read /proc/self/io");
        } else {
            buffer[count] = '\0';
            char *line = buffer;
            while (line != NULL && *line != '\0') {
                char *next = strchr(line, '\n');
                if (next != NULL) {
                    *next++ = '\0';
                }
                char name[64];
                unsigned long long value;
                if (sscanf(line, "%63[^:]: %llu", name, &value) == 2) {
                    for (unsigned int i = 0; i < 7; i++) {
                        if (strcmp(name, names[i]) == 0) {
                            values[i] = value;
                            found |= 1U << i;
                        }
                    }
                }
                line = next;
            }
            if (found != 127) {
                status = "invalid_proc_io";
                snprintf(error, sizeof(error), "incomplete /proc/self/io");
            }
        }
        syscall(SYS_close, input);
    }
    char report[1024];
    int length = snprintf(report, sizeof(report),
        "io_collector_version=4\nio_child_pid=%ld\nio_status=%s\nio_error=%s\n"
        "io_rchar=%llu\nio_wchar=%llu\nio_syscr=%llu\nio_syscw=%llu\n"
        "io_read_bytes=%llu\nio_write_bytes=%llu\nio_cancelled_write_bytes=%llu\n",
        (long)target_pid, status, error, values[0], values[1], values[2], values[3],
        values[4], values[5], values[6]);
    if (length > 0 && (size_t)length < sizeof(report)) {
        int output = (int)syscall(SYS_openat, AT_FDCWD, report_path,
                                  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (output >= 0) {
            size_t written = 0;
            while (written < (size_t)length) {
                ssize_t count = (ssize_t)syscall(SYS_write, output, report + written,
                                               (size_t)length - written);
                if (count < 0 && errno == EINTR) {
                    continue;
                }
                if (count <= 0) {
                    break;
                }
                written += (size_t)count;
            }
            syscall(SYS_close, output);
        }
    }
    free(report_path);
    report_path = NULL;
    errno = saved_errno;
}
