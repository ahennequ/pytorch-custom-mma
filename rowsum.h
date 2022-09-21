#pragma once

template <typename scalar_t, typename warp_tile_t>
struct rowsum_accumulator {
    static constexpr int N_tile = warp_tile_t::N_tile;
    static constexpr int M_tile = warp_tile_t::M_tile;

    float acc;

    __device__ void zero() {
        acc = 0;
    }

    template<typename shared_fragment>
    __device__ void add(shared_fragment& smem) {
        if (threadIdx.x < N_tile) {
            #pragma unroll
            for (int i = 0; i < M_tile; i++) {
                acc += smem(threadIdx.x, i);
            }
        }
    }

    __device__ void divide(scalar_t* smem, warp_tile_t& mma) {
        if (threadIdx.x < N_tile) smem[threadIdx.x] = 1.f / acc;
        __syncthreads();

        mma.pointwise([&](scalar_t el, int, int y) {
            return el * smem[y];
        });
        __syncthreads();
    }

    template<typename accessor>
    __device__ void store(accessor gmem, int tile_y) {
        if (threadIdx.x < N_tile) {
            gmem[threadIdx.x + tile_y] = acc;
        }
    }
};