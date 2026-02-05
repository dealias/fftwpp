#pragma once
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>

/* Instruction State Scrubber: prepare multicore CPUs for maximum throughput
 *
 * Flood the CPU with multiple independent instruction streams
 * (via fork/exec of compiler processes) to saturate the front-end pipelines:
 * DSB (Decoded Stream Buffer), MITE, and LSD.
 * Exercise branch predictors with thousands of branches and indirect calls
 * across cores.
 * Stress memory prefetchers and caches with multiple short-lived processes.
 * Increase IPC, reduce L2/LLC stalls, and produce a persistent
 * high-throughput microarchitectural state.
 *
            +-----------------------+
            |    Each CPU Core      |
            |   +-------------+     |
  forked -->|   | MITE/LSD/DSB|<-- Diverse compiler processes
  process   |   +-------------+     |
            |   | Branch Pred. |    |
            |   | L1 I/D Cache |    |
            |   | L2/L3 Cache  |    |
            +-----------------------+
 */

class CpuPrimer {
public:
  static void prime(int storm_strength = 48) {
    static constexpr char payload[] = "#include <random>\nstd::mt19937 gen;\n";
    constexpr size_t payload_len = sizeof(payload) - 1;

    std::vector<pid_t> children;
    children.reserve(storm_strength);

    for (int i = 0; i < storm_strength; ++i) {
      int pipefd[2];
      if (pipe(pipefd) != 0) break;

      pid_t pid = fork();
      if (pid == 0) { // Child process
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        close(pipefd[0]);
        freopen("/dev/null", "w", stdout);
        freopen("/dev/null", "w", stderr);
        execlp("c++", "c++", "-x", "c++", "-", "-O3", "-march=native", "-c", "-o", "/dev/null", nullptr);
        _exit(127);
      } else if (pid > 0) { // Parent process
        close(pipefd[0]);
        write(pipefd[1], payload, payload_len);
        close(pipefd[1]);
        children.push_back(pid);
      }
    }

    // Wait for all children to complete
    for (pid_t pid : children)
      waitpid(pid, nullptr, 0);
  }
};
