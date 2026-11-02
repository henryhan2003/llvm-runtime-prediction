// Small fixed-memory compute load. Duty is calibrated outside target timing.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

static void checked(cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << cudaGetErrorString(status) << std::endl;
    std::exit(2);
  }
}

__global__ void compute(float *output, int iterations) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  float value = 0.25f + (index % 1024) * 0.0001f;
  for (int i = 0; i < iterations; ++i) {
    value = fmaf(value, 0.99991f, 0.00009f);
  }
  output[index] = value;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: worker DUTY_FILE READY_FILE TIMEOUT_SECONDS PERIOD_MS\n";
    return 2;
  }
  const double timeout = std::stod(argv[3]);
  const double period_ms = std::stod(argv[4]);
  if (!(timeout > 0 && timeout <= 3600 && period_ms >= 20 && period_ms <= 100)) return 2;
  checked(cudaSetDevice(0));
  cudaDeviceProp properties;
  checked(cudaGetDeviceProperties(&properties, 0));
  const int blocks = properties.multiProcessorCount * 8;
  const size_t bytes = static_cast<size_t>(blocks) * 256 * sizeof(float);
  float *output = nullptr;
  checked(cudaMalloc(reinterpret_cast<void **>(&output), bytes));
  cudaEvent_t start, end;
  checked(cudaEventCreate(&start));
  checked(cudaEventCreate(&end));
  int iterations = 4096;
  float elapsed_ms = 0;
  for (int attempt = 0; attempt < 3; ++attempt) {
    checked(cudaEventRecord(start));
    compute<<<blocks, 256>>>(output, iterations);
    checked(cudaGetLastError());
    checked(cudaEventRecord(end));
    checked(cudaEventSynchronize(end));
    checked(cudaEventElapsedTime(&elapsed_ms, start, end));
    if ((elapsed_ms >= 0.5f && elapsed_ms <= 2.0f) || attempt == 2) break;
    const double ratio = 1.0 / std::max(0.001, static_cast<double>(elapsed_ms));
    iterations = static_cast<int>(std::max(64.0, std::min(1000000.0, iterations * ratio)));
  }
  const std::string temporary = std::string(argv[2]) + ".tmp";
  {
    std::ofstream ready(temporary);
    ready << "{\"pid\":" << getpid() << ",\"buffer_bytes\":" << bytes
          << ",\"blocks\":" << blocks << ",\"threads\":256,\"iterations\":" << iterations
          << ",\"calibration_kernel_ms\":" << elapsed_ms << "}";
    if (!ready) return 2;
  }
  if (std::rename(temporary.c_str(), argv[2]) != 0) return 2;
  using Clock = std::chrono::steady_clock;
  const auto deadline = Clock::now() + std::chrono::duration<double>(timeout);
  while (Clock::now() < deadline) {
    double duty = -1;
    { std::ifstream control(argv[1]); control >> duty; }
    if (!(duty >= 0 && duty <= 0.85)) return 2;
    const auto cycle = Clock::now();
    const auto active_end = cycle + std::chrono::duration<double, std::milli>(period_ms * duty);
    while (Clock::now() < active_end && Clock::now() < deadline) {
      compute<<<blocks, 256>>>(output, iterations);
      checked(cudaGetLastError());
      checked(cudaDeviceSynchronize());
    }
    std::this_thread::sleep_until(cycle + std::chrono::duration<double, std::milli>(period_ms));
  }
  checked(cudaEventDestroy(start));
  checked(cudaEventDestroy(end));
  checked(cudaFree(output));
  return 0;
}
