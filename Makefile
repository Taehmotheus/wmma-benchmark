CCSYCL = icpx
CCNVIDIA = nvcc
CCAMD = hipcc

LINK_CUBLAS = -lcublas
LINK_ROCBLAS = -lrocblas

SYCL_ARCH = -fsycl-targets=
SYCL_AMD_ARCH = $(SYCL_ARCH)amd_gpu_gfx90a
SYCL_NVIDIA_ARCH = $(SYCL_ARCH)nvidia_gpu_sm80

dgemm_cublas.x: src/cuda_dgemm_cuBLAS.cu
	$(CCNVIDIA) -Iinclude $(LINK_CUBLAS) src/cuda_dgemm_cuBLAS.cu -o dgemm_cublas.x

dgemm_rocblas.x: src/rocm_dgemm_rocBLAS.cpp
	$(CCAMD) -Iinclude $(LINK_ROCBLAS) src/rocm_dgemm_rocBLAS.cpp -o dgemm_rocblas.x


clean:
	rm *.x
