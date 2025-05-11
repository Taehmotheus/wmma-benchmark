#include "sycl/sycl.hpp"
#include <iomanip>
#include <vector>

using namespace sycl::ext::oneapi::experimental::matrix;
using namespace sycl;

using bfloat16 = sycl::ext::oneapi::bfloat16;
using matMul = double;
using matAcc = double;

std::vector<sycl::event> work_events;

double get_kernel_runtime() {
    double kernel_runtime = 0.0;
    for (sycl::event work_event : work_events) {
        const double timesteps_per_second = 1.0e9;
        double start =
            double(work_event.get_profiling_info<sycl::info::event_profiling::command_start>()) /
            timesteps_per_second;
        double end =
            double(work_event.get_profiling_info<sycl::info::event_profiling::command_end>()) /
            timesteps_per_second;
        kernel_runtime += end - start;
    }
    return kernel_runtime;
}

// Convert SYCL query type to std::string
std::string to_string(matrix_type type) {
    switch (type) {
    case matrix_type::bf16:
        return "bf16";
    case matrix_type::fp16:
        return "fp16";
    case matrix_type::tf32:
        return "tf32";
    case matrix_type::fp32:
        return "fp32";
    case matrix_type::fp64:
        return "fp64";
    case matrix_type::sint8:
        return "sint8";
    case matrix_type::sint16:
        return "sint16";
    case matrix_type::sint32:
        return "sint32";
    case matrix_type::sint64:
        return "sint64";
    case matrix_type::uint8:
        return "uint8";
    case matrix_type::uint16:
        return "uint16";
    case matrix_type::uint32:
        return "uint32";
    case matrix_type::uint64:
        return "uint64";
    default:
        return "Unknown";
    }
}

bool verify_result(const matMul *A, const matMul *B, const matAcc *C_device, size_t M, size_t N,
                   size_t K, double tol = 1e-6) {
    bool correct = true;
    for (size_t row = 0; row < M; row++) {
        for (size_t col = 0; col < N; col++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            double diff = std::abs(C_device[row * N + col] - sum);
            if (diff > tol) {
                std::cerr << "Mismatch at (" << row << ", " << col << "): "
                          << "Expected " << sum << ", but got " << C_device[row * N + col]
                          << " (diff = " << diff << ")\n";
                correct = false;
                return correct; // early exit
            }
        }
    }
    return correct;
}

int main() {
    sycl::device device(sycl::gpu_selector_v);

    std::cout
        << "\n ----------------------------------------------------------------------------- \n"
        << std::endl;
    std::cout << "Running on: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Device max_compute_units: "
              << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Device local mem: " << device.get_info<sycl::info::device::local_mem_size>()
              << std::endl;
    std::cout << "Max Work Group Size: "
              << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;

    auto sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>();

    // sub-group size
    std::cout << "Max Sub Group Sizes: ";
    for (auto size : sub_group_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // query supported combinations of the device matrix/tensor core
    auto combinations =
        device.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    std::cout
        << "\n ----------------------------------------------------------------------------- \n"
        << std::endl;

    std::cout << "Supported matrix combination:\n";
    for (const auto &comb : combinations) {
        std::cout
            << "-----------------------------------------------------------------------------\n";
        std::cout << "Max M: " << comb.max_msize << ", Max N: " << comb.max_nsize
                  << ", Max K: " << comb.max_ksize << "\n";
        std::cout << "M: " << comb.msize << ", N: " << comb.nsize << ", K: " << comb.ksize << "\n";
        std::cout << "A-Typ: " << to_string(comb.atype) << ", B-Typ: " << to_string(comb.btype)
                  << ", C-Typ: " << to_string(comb.ctype) << ", D-Typ: " << to_string(comb.dtype)
                  << "\n";
    }
    std::cout
        << "\n ----------------------------------------------------------------------------- \n"
        << std::endl;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};

    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};

    int n_repeats = 10;

    for (size_t sizes : mat_sizes) {
        // defining sizes Dmxn = Amxk * Bkxm + Cmxn
        size_t M = sizes;
        size_t N = sizes;
        size_t K = sizes;

        // warp/wavefront size
        constexpr size_t SG_SZ = 64;

        // supported device matrix size
        constexpr size_t TM = 16;
        constexpr size_t TN = 16;
        constexpr size_t TK = 4;

        // calculation of how many work-groups will exist in every dimension
        size_t NDRangeM = M / TM;
        size_t NDRangeN = N / TN;

        // allocate memory
        matMul *memA = malloc_shared<matMul>(M * K, q);
        matMul *memB = malloc_shared<matMul>(K * N, q);
        matAcc *memC = malloc_shared<matAcc>(M * N, q);

        // fill memory with data
        for (size_t i = 0; i < M * K; ++i) {
            memA[i] = (matMul)1;
        }
        for (size_t i = 0; i < K * N; ++i) {
            memB[i] = (matMul)1;
        }

        for (int i = 0; i < n_repeats; i++) {
            sycl::event work_event = q.submit([&](sycl::handler &cgh) {
                // ensures that the data is in global memory
                auto pA = address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(memA);
                auto pB = address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(memB);
                auto pC = address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(memC);

                cgh.parallel_for(
                    sycl::nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, SG_SZ}),
                    [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SZ)]]

                    {
                        const auto sg_startx = item.get_global_id(0) - item.get_local_id(0);
                        const auto sg_starty = item.get_global_id(1) - item.get_local_id(1);
                        sub_group sg = item.get_sub_group();

                        joint_matrix<sub_group, matMul, use::a, TM, TK, layout::row_major> subA;
                        joint_matrix<sub_group, matMul, use::b, TK, TN, layout::row_major> subB;
                        joint_matrix<sub_group, matAcc, use::accumulator, TM, TN> subC;
                        joint_matrix_fill(sg, subC, 0);

                        for (int k = 0; k < K; k += TK) {

                            joint_matrix_load(sg, subA, pA + sg_startx * TM * K + k, K);
                            joint_matrix_load(sg, subB, pB + k * N + sg_starty / SG_SZ * TN, N);
                            joint_matrix_mad(sg, subC, subA, subB, subC);
                        }
                        joint_matrix_store(sg, subC,
                                           pC + sg_startx * TM * N + sg_starty / SG_SZ * TN, N,
                                           layout::row_major);
                    });
            });
            q.wait();
            work_events.push_back(work_event);
        }

        // TODO verification of the results (No need for now because of the bad performance)

        double avg_time = get_kernel_runtime() / n_repeats;
        double gflops = 2.0 * N * N * N * 1e-9 / avg_time;

        std::cout << "For: " << sizes << " " << "Kerneltime: " << avg_time << " s" << std::endl;

        std::cout << "GFLOPS/s : " << gflops << "\n";

        work_events.clear();
    }

    return 0;
}
