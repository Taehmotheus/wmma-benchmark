#include <chrono>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <rocblas/rocblas.h>
#include <vector>

void random_init(double *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = static_cast<double>(rand()) / RAND_MAX * 20.0 - 10.0;
    }
}

int main() {
    int device_id;

    // Get the current device being used
    hipGetDevice(&device_id);

    std::cout << "Running on GPU: " << device_id << std::endl;

    auto startWall = std::chrono::high_resolution_clock::now();
    int mat_sizes[] = {128,  256,  512,  1024,
                       2048, 4096, 8192, 16384 /*, 32768  , 65536 , 131072, 262144*/};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

    for (int i = 0; i < n_sizes; i++) {
        int N = mat_sizes[i];
        int size = N * N;

        // Host memory
        std::vector<double> h_A(size), h_B(size), h_C(size);
        random_init(h_A.data(), N);
        random_init(h_B.data(), N);

        // Device memory
        double *d_A, *d_B, *d_C;
        hipMalloc(&d_A, size * sizeof(double));
        hipMalloc(&d_B, size * sizeof(double));
        hipMalloc(&d_C, size * sizeof(double));

        hipMemcpy(d_A, h_A.data(), size * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), size * sizeof(double), hipMemcpyHostToDevice);

        // Create rocBLAS handle
        rocblas_handle handle;
        rocblas_create_handle(&handle);

        // Warmup
        double alpha = 1.0, beta = 1.0;
        rocblas_dgemm_64(handle, rocblas_operation_none, rocblas_operation_none, N, N, N, &alpha,
                         d_A, N, d_B, N, &beta, d_C, N);
        hipDeviceSynchronize();

        // Initialize GPU Events for precise timing
        hipEvent_t startEvent, stopEvent;
        hipEventCreate(&startEvent);
        hipEventCreate(&stopEvent);

        // Timed runs
        const int n_repeats = 1;
        hipEventRecord(startEvent, 0); // Start Event
        for (int j = 0; j < n_repeats; j++) {
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, N, N, N, &alpha,
                          d_A, N, d_B, N, &beta, d_C, N);
        }
        hipDeviceSynchronize();
        hipEventRecord(stopEvent, 0); // Stop Event

        // Synchronize and get elapsed time
        hipEventSynchronize(stopEvent);
        float elapsedTime;
        hipEventElapsedTime(&elapsedTime, startEvent, stopEvent);

        // Compute the average time and GFLOPS
        double avg_time = elapsedTime / n_repeats / 1000.0; // in seconds
        double tflops = 2.0 * N * N * N * 1e-9 / avg_time;

        std::cout << "N: " << std::setw(6) << N << " | Time: " << std::fixed << std::setprecision(6)
                  << avg_time << " s"
                  << " | GFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";

        // Cleanup
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        rocblas_destroy_handle(handle);
        hipEventDestroy(startEvent);
        hipEventDestroy(stopEvent);
    }
    auto endWall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedWall = endWall - startWall;
    std::cout << elapsedWall.count() << " s";
    return 0;
}
