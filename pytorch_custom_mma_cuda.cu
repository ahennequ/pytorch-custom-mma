#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <c10/cuda/CUDAGuard.h>
#include <type_traits>

#include <torch/extension.h>

#include "MMA.h"
#include "mem.h"
#include "rowsum.h"

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

template <typename scalar_t>
__global__ void forward_kernel(
    const PackedAccessor<scalar_t, 3> Q,
    const PackedAccessor<scalar_t, 3> K,
    const PackedAccessor<scalar_t, 3> V,
          PackedAccessor<scalar_t, 3> O,
          PackedAccessor<scalar_t, 2> l,
    const float scale
) {
    const int batch = blockIdx.y;

    const int N = Q.size(1);
    const int M = K.size(1);
    const int D = Q.size(2);

    const int tile_w = M / mma::warp_tile<scalar_t>::M_tile;
    const int tile_y = blockIdx.x;

    extern __shared__ char _shared_mem[];

    mma::warp_tile<scalar_t> QK_mma; // 32x16 tile per warp in registers -> process 64x64 with the block
    mma::warp_tile<scalar_t> out_mma;
    rowsum_accumulator<scalar_t, mma::warp_tile<scalar_t>> L_acc;
    
    mem::shared_fragment<scalar_t> Q_sm{_shared_mem, D, mma::warp_tile<scalar_t>::N_tile};
    mem::shared_fragment<scalar_t> K_sm{Q_sm.next(), D, mma::warp_tile<scalar_t>::M_tile};
    mem::shared_fragment<scalar_t> C_sm{K_sm.next(), mma::warp_tile<scalar_t>::N_tile, mma::warp_tile<scalar_t>::M_tile};

    out_mma.zero();
    L_acc.zero();

    Q_sm.load_transpose(Q[batch], 0, tile_y);
    for (int tile_x = 0; tile_x < tile_w; tile_x++) {
        K_sm.load_transpose(K[batch], 0, tile_x);

        __syncthreads();

        QK_mma.zero();

        for (int d = 0; d < D; d += mma::warp_tile<scalar_t>::K_tile) {
            QK_mma.mma(Q_sm.smem, K_sm.smem, d);
        }

        QK_mma.pointwise([&](scalar_t el) -> scalar_t {
            return expf(scale * el - scale); 
        });

        QK_mma.store_transpose(C_sm.smem);

        __syncthreads();

        // Second matmul:
        K_sm.load(V[batch], 0, tile_x); // reuse K shared mem for V

        __syncthreads();

        for (int j = 0; j < mma::warp_tile<scalar_t>::M_tile; j += mma::warp_tile<scalar_t>::K_tile) {
            out_mma.mma(C_sm.smem, K_sm.smem, j);
            L_acc.add(C_sm.smem, j);
        }

        __syncthreads();
    }

    L_acc.store(l[batch], tile_y);
    L_acc.divide(C_sm.smem, out_mma);

    out_mma.store(C_sm.smem);
    __syncthreads();

    C_sm.store(O[batch], 0, tile_y);
}

std::vector<at::Tensor> mma_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));

    const int batch = Q.size(0);
    const int N = Q.size(1);
    const int M = K.size(1);
    const int D = Q.size(2);

    auto options = torch::TensorOptions().device(device_of(Q)).dtype(Q.scalar_type());
    auto O = at::empty({batch, N, D}, options);
    auto l = at::empty({batch, N}, options);

    const dim3 threads_per_block(256);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "forward_cosine_sim_attention_forward", ([&] {
        const dim3 blocks(N / mma::warp_tile<scalar_t>::N_tile, batch);
        const unsigned shared_mem_size = (mma::warp_tile<scalar_t>::N_tile * D +
                                          mma::warp_tile<scalar_t>::M_tile * D +
                                          mma::warp_tile<scalar_t>::N_tile * mma::warp_tile<scalar_t>::M_tile) * sizeof(scalar_t);
        forward_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            ACCESSOR(Q, 3, scalar_t),
            ACCESSOR(K, 3, scalar_t),
            ACCESSOR(V, 3, scalar_t),
            ACCESSOR(O, 3, scalar_t),
            ACCESSOR(l, 2, scalar_t),
            scale
        );
    }));

    // handle error
    //cudaDeviceSynchronize();
    //CHECK_LAST_CUDA_ERROR();

    return { O, l };
}

// bind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mma_forward, "MMA Forward");
    // m.def("backward", &mma_backward, "MMA Backward");
}
