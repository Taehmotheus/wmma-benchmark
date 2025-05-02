#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/common_utils.hpp"

int main() {
    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int n_sizes = mat_sizes.size();

    // Loop over all Matrix sizes
    for (int i = 0; i < n_sizes; i++) {
        int N = mat_sizes[i];
        int size = N * N;

        // Initialize memory on host
        std::vector<double> h_A(size), h_B(size), h_C(size);
        random_matrix<double>(h_A.data(), N);
        random_matrix<double>(h_B.data(), N);

        // Allocate memory on device and copy
        double *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size * sizeof(double));
        cudaMalloc(&d_B, size * sizeof(double));
        cudaMalloc(&d_C, size * sizeof(double));

        cudaMemcpy(d_A, h_A.data(), size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), size * sizeof(double), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Warmup
        double alpha = random_value<double>(), beta = random_value<double>();
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C,
                    N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Timed runs
        const int n_repeats = 10;
        cudaEventRecord(start);

        for (int j = 0; j < n_repeats; j++) {
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta,
                        d_C, N);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        double avg_time = milliseconds / n_repeats;
        double tflops = 2.0 * N * N * N * 1e-9 / avg_time;

        std::cout << "N: " << std::setw(6) << N << " | Time: " << std::fixed << std::setprecision(6)
                  << avg_time << " s"
                  << " | TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

    return 0;
}
