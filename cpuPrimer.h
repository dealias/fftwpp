#pragma once
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>

/* cpu_primer.h: Instruction State Scrubber
 *
 * Eliminate latency on high-performance Intel architectures (Alder Lake+) to
 * ensure consistent benchmark timing by evicting stale code from the CPU
 * instruction cache and branch target buffer associated with OS kernel tasks.
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
