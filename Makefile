CCSYCL = icpx
CCNVIDIA = nvcc
CCAMD = hipcc

LINK_CUBLAS = -lcublas
LINK_ROCBLAS = -lrocblas

SYCL_FLAGS = -fsycl

SYCL_ARCH = -fsycl-targets=
SYCL_AMD_ARCH = $(SYCL_ARCH)amd_gpu_gfx90a
SYCL_NVIDIA_ARCH = $(SYCL_ARCH)nvidia_gpu_sm_80

dgemm_cublas.x: src/cuda_dgemm_cuBLAS.cu
	$(CCNVIDIA) -Iinclude $(LINK_CUBLAS) src/cuda_dgemm_cuBLAS.cu -o dgemm_cublas.x

dgemm_rocblas.x: src/rocm_dgemm_rocBLAS.cpp
	$(CCAMD) -Iinclude $(LINK_ROCBLAS) src/rocm_dgemm_rocBLAS.cpp -o dgemm_rocblas.x

dgemm_batched_rocblas.x: src/rocm_dgemm_batched_rocBLAS.cpp
	$(CCAMD) -Iinclude $(LINK_ROCBLAS) src/rocm_dgemm_batched_rocBLAS.cpp -o dgemm_rocblas.x

dgemm_naive_wmma_sycl_rocm.x: src/rocm_joint_matrix_FP64.cpp
	$(CCSYCL) $(SYCL_FLAGS) -Iinclude $(SYCL_AMD_ARCH) src/rocm_joint_matrix_FP64.cpp -o dgemm_rocblas.x

dgemm_naive_wmma_sycl_cuda.x: src/cuda_joint_matrix_FP64.cpp
	$(CCSYCL) $(SYCL_FLAGS) -Iinclude $(SYCL_NVIDIA_ARCH) src/cuda_joint_matrix_FP64.cpp -o dgemm_rocblas.x

dgemm_naive_wmma_cuda.x: src/cuda_wmma_FP64.cu
	$(CCNVIDIA) -Iinclude src/cuda_wmma_FP64.cu -o dgemm_rocblas.x -arch=sm_80 -lcublas

dgemm_naive_wmma_amd.x: src/amd_wmma_FP64.cpp
	$(CCAMD) -std=c++17 -lrocblas src/amd_wmma_FP64.cpp -o dgemm_rocblas.x --offload-arch=gfx90a


clean:
	rm *.x
