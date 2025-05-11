#include <cassert>
#include <cmath>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <rocblas/rocblas.h>
#include <rocwmma/rocwmma.hpp>
#include <vector>

using namespace rocwmma;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 4;

__global__ void wmma_kernel(double *a, double *b, double *c, int M, int N, int K) {
    using AFrag = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, double, row_major>;
    using BFrag = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, double, row_major>;
    using CFrag = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, double>;

    int warpM = blockIdx.x;
    int warpN = blockIdx.y;

    AFrag a_frag;
    BFrag b_frag;
    CFrag c_frag;

    fill_fragment(c_frag, 0.0);

    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        load_matrix_sync(a_frag, a + aRow * K + aCol, K);
        load_matrix_sync(b_frag, b + bRow * N + bCol, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    store_matrix_sync(c + cRow * N + cCol, c_frag, N, mem_row_major);
}

int main() {
    const int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    const int numSizes = sizeof(sizes) / sizeof(int);

    for (int idx = 0; idx < numSizes; ++idx) {
        int N = sizes[idx];

        size_t bytes = N * N * sizeof(double);
        double *A, *B, *C, *C_ref;
        hipMalloc(&A, bytes);
        hipMalloc(&B, bytes);
        hipMalloc(&C, bytes);
        hipMalloc(&C_ref, bytes);

        // Initialize matrices
        std::vector<double> hA(N * N), hB(N * N), hC(N * N, 0), hC_ref(N * N, 0);
        for (int i = 0; i < N * N; ++i) {
            hA[i] = (rand() % 20) - 10;
            hB[i] = (rand() % 20) - 10;
        }

        hipMemcpy(A, hA.data(), bytes, hipMemcpyHostToDevice);
        hipMemcpy(B, hB.data(), bytes, hipMemcpyHostToDevice);
        hipMemcpy(C, hC.data(), bytes, hipMemcpyHostToDevice);
        hipMemcpy(C_ref, hC_ref.data(), bytes, hipMemcpyHostToDevice);

        dim3 gridDim(N / WMMA_M, N / WMMA_N);
        dim3 blockDim(64);

        hipLaunchKernelGGL(wmma_kernel, gridDim, blockDim, 0, 0, A, B, C, N, N, N);
        hipDeviceSynchronize();

        // Timing
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start);

        for (int i = 0; i < 10; ++i) {
            hipLaunchKernelGGL(wmma_kernel, gridDim, blockDim, 0, 0, A, B, C, N, N, N);
        }

        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);

        double elapsed = ms / 1000.0;
        double gflops = 2.0 * N * N * N * 10 / (elapsed * 1e9);

        // Validate with rocBLAS
        rocblas_handle handle;
        rocblas_create_handle(&handle);

        const double alpha = 1.0;
        const double beta = 0.0;

        rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, N, N, N, &alpha, B, N,
                      A, N, &beta, C_ref, N);
        hipDeviceSynchronize();

        hipMemcpy(hC.data(), C, bytes, hipMemcpyDeviceToHost);
        hipMemcpy(hC_ref.data(), C_ref, bytes, hipMemcpyDeviceToHost);

        for (int i = 0; i < N * N; ++i) {
            if (fabs(hC[i] - hC_ref[i]) > 1e-8) {
                std::cerr << "Mismatch at index " << i << ": " << hC[i] << " vs " << hC_ref[i]
                          << "\n";
                assert(false);
            }
        }

        std::cout << "N=" << N << " | Time=" << elapsed / 10 << "s | GFLOPS=" << gflops << "\n";

        hipFree(A);
        hipFree(B);
        hipFree(C);
        hipFree(C_ref);
        rocblas_destroy_handle(handle);
    }

    return 0;
}
