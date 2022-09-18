#pragma once

template <typename scalar_t, typename warp_tile_t>
struct rowsum_accumulator {
    static constexpr int N_thread = warp_tile_t::N_thread;
    static constexpr int N_warp   = warp_tile_t::N_warp;
    static constexpr int N_tile   = warp_tile_t::N_tile;

    float acc;

    __device__ void zero() {
        acc = 0;
    }

    __device__ void add(scalar_t* smem, int d) {
        if (threadIdx.x < N_tile) {
            #pragma unroll
            for (int i = 0; i < warp_tile_t::K_tile; i++) {
                acc += smem[threadIdx.x + (d + i) * N_tile];
            }
        }
    }

    __device__ void divide(scalar_t* smem, warp_tile_t& mma) {
        if (threadIdx.x < N_tile) smem[threadIdx.x] = 1.f / acc;
        __syncthreads();

        mma.rowwise([&](scalar_t el, int i) {
            return el * smem[i];
        });
        __syncthreads();
    }

    template<typename accessor>
    __device__ void store(accessor gmem, int tile_y) {
        if (threadIdx.x < N_tile) {
            gmem[threadIdx.x + tile_y * N_tile] = acc;
        }
    }
};