#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <mma.h>

#define cublas_check(status)                                                                       \
    {                                                                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            std::cerr << "cuBLAS Error" << std::endl;                                              \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

using namespace nvcuda;

const int WMMA_M = 8;
const int WMMA_N = 8;
const int WMMA_K = 4;

__global__ void wmma_kernel(double *d_A_ptr, double *d_B_ptr, double *d_C_ptr, int M, int N,
                            int K) {

    int warpM = blockIdx.x; //(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> c_frag;

    fill_fragment(c_frag, 0.0);

    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        wmma::load_matrix_sync(a_frag, &d_A_ptr[aRow * K + aCol], K);
        wmma::load_matrix_sync(b_frag, &d_B_ptr[bRow * N + bCol], N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    wmma::store_matrix_sync(&d_C_ptr[cRow * N + cCol], c_frag, N, wmma::mem_row_major);
}

int main() {
    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Store time and GFLOPS
    double tgemm_time[n_sizes];
    double tgemm_gflops[n_sizes];

    for (int mat_size = 0; mat_size < n_sizes; mat_size++) {
        int n = mat_sizes[mat_size];

        double *A_ptr = new double[n * n];
        double *B_ptr = new double[n * n];
        double *C_ptr = new double[n * n];
        double *C_ptr_verification = new double[n * n];

        for (size_t i = 0; i < n * n; i++) {
            A_ptr[i] = (double)(rand() % (-10 - 10 + 1) + 10);
            B_ptr[i] = (double)(rand() % (-10 - 10 + 1) + 10);
            C_ptr_verification[i] = 0.0;
        }

        double *dA_ptr, *dB_ptr, *dC_ptr, *dC_ptr_verification;

        cudaMalloc((void **)&dA_ptr, n * n * sizeof(double));
        cudaMalloc((void **)&dB_ptr, n * n * sizeof(double));
        cudaMalloc((void **)&dC_ptr, n * n * sizeof(double));
        cudaMalloc((void **)&dC_ptr_verification, n * n * sizeof(double));

        cudaMemcpy(dA_ptr, A_ptr, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dB_ptr, B_ptr, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dC_ptr, C_ptr, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dC_ptr_verification, C_ptr_verification, n * n * sizeof(double),
                   cudaMemcpyHostToDevice);

        dim3 dim_block(32, 1);
        dim3 dim_grid(n / WMMA_M, n / WMMA_N);

        wmma_kernel<<<dim_grid, dim_block>>>(dA_ptr, dB_ptr, dC_ptr, n, n, n);

        cudaMemcpy(C_ptr, dC_ptr, n * n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaEventRecord(beg);
        for (int n_runs = 0; n_runs < 10; n_runs++) {
            wmma_kernel<<<dim_grid, dim_block>>>(dA_ptr, dB_ptr, dC_ptr, n, n, n);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;

        tgemm_time[mat_size] = (elapsed_time) / 10;
        tgemm_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time);

        // Create and initialize cuBLAS handle
        cublasHandle_t handle;
        cublas_check(cublasCreate(&handle));

        // Perform matrix multiplication: C = A * B
        double alpha = 1;
        double beta = 0;
        cublas_check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB_ptr,
                                  CUDA_R_64F, n, dA_ptr, CUDA_R_64F, n, &beta, dC_ptr_verification,
                                  CUDA_R_64F, n, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cudaDeviceSynchronize();

        cudaMemcpy(C_ptr_verification, dC_ptr_verification, n * n * sizeof(double),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (fabs(C_ptr[i * n + j] - C_ptr_verification[i * n + j]) > 1e-8) {
                    std::cerr << "Assertion failed for " << "row number: " << i
                              << ", col number: " << j << ".\n"
                              << "Absolute Difference: "
                              << fabs(C_ptr[i * n + j] - C_ptr_verification[i * n + j]) << "\n";
                    assert(fabs(C_ptr[i * n + j] - C_ptr_verification[i * n + j]) < 1e-8 &&
                           "Assertion failed!");
                }
            }
        }

        delete[] A_ptr;
        delete[] B_ptr;
        delete[] C_ptr;
        delete[] C_ptr_verification;
        cudaFree(dA_ptr);
        cudaFree(dB_ptr);
        cudaFree(dC_ptr);
        cudaFree(dC_ptr_verification);
    }

    // Printing Results
    std::cout << "Matrix Size: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << mat_sizes[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "tGeMM Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << tgemm_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "tGeMM GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << tgemm_gflops[mat_size] << " ";
    std::cout << "\n \n";

    return 0;
}