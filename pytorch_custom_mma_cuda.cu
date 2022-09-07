#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>

#include "MMA.h"
#include "mem.h"

#define CHECK_LAST_CUDA_ERROR() check(__FILE__, __LINE__)
void check(const char* file, const int line) {
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

#define ACCESSOR(x, n, type) x.packed_accessor32<type, n, torch::RestrictPtrTraits>()

// type alias

template <typename scalar_t, int dims>
using PackedAccessor = torch::PackedTensorAccessor32<scalar_t, dims, torch::RestrictPtrTraits>;

__global__ void forward_kernel(
    const PackedAccessor<float, 3> A,
    const PackedAccessor<float, 3> B,
    const PackedAccessor<float, 3> V,
          PackedAccessor<float, 3> C,
    const float scale
) {
    const int batch = blockIdx.y;

    const int N = A.size(1);
    const int M = B.size(1);
    const int D = A.size(2);

    const int tile_w = M / mma::warp_tile::M_tile;
    const int tile_y = blockIdx.x;

    extern __shared__ float _shared_mem[];

    mma::warp_tile AB_mma; // 32x16 tile per warp in registers -> process 64x64 with the block
    mma::warp_tile out_mma; 
    
    mem::shared_fragment<float> A_sm{_shared_mem, D, mma::warp_tile::N_tile};
    mem::shared_fragment<float> B_sm{A_sm.next(), D, mma::warp_tile::M_tile};
    mem::shared_fragment<float> C_sm{B_sm.next(), mma::warp_tile::N_tile, mma::warp_tile::M_tile};

    out_mma.zero();

    A_sm.load_transpose(A[batch], 0, tile_y);
    for (int tile_x = 0; tile_x < tile_w; tile_x++) {
        B_sm.load_transpose(B[batch], 0, tile_x);

        __syncthreads();

        AB_mma.zero();

        for (int d = 0; d < D; d++) {
            AB_mma.mma(A_sm.smem, B_sm.smem, d);
        }

        AB_mma.pointwise([&](float el) {
            return expf(scale * el); 
        });

        AB_mma.store_transpose(C_sm.smem);

        __syncthreads();

        // Second matmul:
        B_sm.load(V[batch], 0, tile_x);

        __syncthreads();

        for (int d = 0; d < D; d++) {
            out_mma.mma(C_sm.smem, B_sm.smem, d);
        }

        __syncthreads();
    }

    out_mma.store(C_sm.smem);
    __syncthreads();

    C_sm.store(C[batch], 0, tile_y);
}

std::vector<at::Tensor> mma_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor V,
    float scale
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    /*auto C = A.mm(B.t());
    C *= scale;
    return { C };*/

    const int batch = A.size(0);
    const int N = A.size(1);
    const int M = B.size(1);
    const int D = A.size(2);

    auto options = torch::TensorOptions().device(device_of(A)).dtype(torch::kFloat);
    auto C = at::empty({batch, N, D}, options);

    const dim3 threads_per_block(256);
    const dim3 blocks(N / mma::warp_tile::N_tile, batch);
    const unsigned shared_mem_size = (mma::warp_tile::N_tile * D +
                                      mma::warp_tile::M_tile * D +
                                      mma::warp_tile::N_tile * mma::warp_tile::M_tile) * sizeof(float);

    forward_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        ACCESSOR(A, 3, float),
        ACCESSOR(B, 3, float),
        ACCESSOR(V, 3, float),
        ACCESSOR(C, 3, float),
        scale
    );

    // handle error
    //cudaDeviceSynchronize();
    //CHECK_LAST_CUDA_ERROR();

    return { C };
}

// bind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mma_forward, "MMA Forward");
    // m.def("backward", &mma_backward, "MMA Backward");
}
