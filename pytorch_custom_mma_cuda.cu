#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <c10/cuda/CUDAGuard.h>
#include <type_traits>

#include <torch/extension.h>

#include "dispatch.h"
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
    const int QK_dim = Q.size(2);

    using QK_mma_t  = mma::warp_tile<scalar_t>;
    using out_mma_t = mma::warp_tile<scalar_t>;

    using Q_sm_t = mem::shared_fragment<scalar_t, 16, QK_mma_t::N_tile>;
    using K_sm_t = mem::shared_fragment<scalar_t, 16, QK_mma_t::M_tile>;
    using C_sm_t = mem::shared_fragment<scalar_t, QK_mma_t::N_tile, QK_mma_t::M_tile>;

    const int tile_y = blockIdx.x * QK_mma_t::N_tile;

    __shared__ scalar_t _shared_mem[Q_sm_t::size + K_sm_t::size + C_sm_t::size];

    QK_mma_t  QK_mma; // 32x16 tile per warp in registers -> process 64x64 with the block
    out_mma_t out_mma;
    rowsum_accumulator<scalar_t, QK_mma_t> L_acc;
    
    Q_sm_t Q_sm{reinterpret_cast<char*>(_shared_mem)};
    K_sm_t K_sm{Q_sm.next()};
    C_sm_t C_sm{K_sm.next()};

    out_mma.zero();
    L_acc.zero();

    for (int tile_x = 0; tile_x < M; tile_x += QK_mma_t::M_tile) {
        QK_mma.zero();

        for (int k = 0; k < QK_dim; k += K_sm_t::N) {
            Q_sm.load_transpose(Q[batch], k, tile_y);
            K_sm.load_transpose(K[batch], k, tile_x);
            __syncthreads();

            QK_mma.mma(Q_sm, K_sm, 0, 0, K_sm_t::N);
            __syncthreads();
        }

        QK_mma.pointwise([&](scalar_t el, int, int) -> scalar_t {
            return expf(scale * el - scale); 
        });

        QK_mma.store_transpose(C_sm);
        __syncthreads();

        L_acc.add(C_sm);

        // Second matmul:
        for (int k = 0; k < QK_mma_t::M_tile; k += K_sm_t::N) {
            K_sm.load(V[batch], 0, tile_x + k); // reuse K shared mem for V
            __syncthreads();

            out_mma.mma(C_sm, K_sm, k, 0, K_sm_t::N);
            __syncthreads();
        }
    }

    L_acc.store(l[batch], tile_y);
    L_acc.divide(C_sm.smem, out_mma);

    out_mma.store(C_sm);
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
    const int QK_dim = Q.size(2);
    const int V_dim = V.size(2);

    auto options = torch::TensorOptions().device(device_of(Q)).dtype(Q.scalar_type());
    auto O = at::empty({batch, N, V_dim}, options);
    auto l = at::empty({batch, N}, options);

    const dim3 threads_per_block(256);

    AT_TYPE_DISPATCH_SWITCH(Q.scalar_type(), scalar_t, (at::ScalarType::Float, at::ScalarType::Half), (
        VALUE_DISPATCH_SWITCH(V_dim, out_dim, (64), (
            const dim3 blocks(N / mma::warp_tile<scalar_t>::N_tile, batch);
            forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                ACCESSOR(Q, 3, scalar_t),
                ACCESSOR(K, 3, scalar_t),
                ACCESSOR(V, 3, scalar_t),
                ACCESSOR(O, 3, scalar_t),
                ACCESSOR(l, 2, scalar_t),
                scale
            );
        ), ())
    ), ())

    // handle error
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    return { O, l };
}

// bind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mma_forward, "MMA Forward");
    // m.def("backward", &mma_backward, "MMA Backward");
}
