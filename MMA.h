#pragma once

#include<mma.h>

namespace mma {
    template<typename scalar_t>
    struct warp_tile {
        // How much data is processed by a single thread:
        static constexpr int N_thread = 4;
        static constexpr int M_thread = 4;

        // Thread layout within a warp:
        static constexpr int N_warp = 8;
        static constexpr int M_warp = 4;
        static_assert(N_warp * M_warp == 32);

        // Warp layout within a block:
        static constexpr int N_block = 2;
        static constexpr int M_block = 4;
        static_assert(N_block * M_block * N_warp * M_warp == 256); // blockDim.x

        // Dimensions of the tile, in threads:
        static constexpr int N_tile = N_warp * N_block * N_thread;
        static constexpr int M_tile = M_warp * M_block * M_thread;
        static constexpr int K_tile = 1;

        // Registers:
        float A_frag[N_thread];            // N x 1 fragment
        float B_frag[M_thread];            // 1 x M fragment
        float C_frag[N_thread * M_thread]; // N x M fragment

        int warp_x;   // x offset of the warp within the block tile
        int warp_y;   // y offset of the warp within the block tile
        int thread_x; // x offset of the thread within the warp tile
        int thread_y; // y offset of the thread within the warp tile

        __device__ warp_tile() {
            int warp_id = threadIdx.x / 32;
            warp_x = (warp_id % M_block);
            warp_y = (warp_id / M_block);

            int lane_id = threadIdx.x % 32;
            thread_x = warp_x * M_warp * M_thread + lane_id % M_warp;
            thread_y = warp_y * N_warp * N_thread + lane_id / M_warp;
        }

        // Initialize C to all zeros
        __device__ void zero() {
            #pragma unroll
            for (int i = 0; i < N_thread * M_thread; i++) {
                C_frag[i] = 0.f;
            }
        }

        // Performs C = A * B + C
        __device__ void mma(const scalar_t* A_sm_ptr, const scalar_t* B_sm_ptr, int k) {
            // Load a N x 1 fragment of A from shared memory to registers:
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                A_frag[i] = A_sm_ptr[i * N_warp + thread_y + k * N_tile];
            }

            // Load a 1 x M fragment of B from shared memory to registers:
            #pragma unroll
            for (int i = 0; i < M_thread; i++) {
                B_frag[i] = B_sm_ptr[i * M_warp + thread_x + k * M_tile];
            }

            // Compute:
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread ; j++) {
                    C_frag[i * M_thread + j] += A_frag[i] * B_frag[j];
                }
            }
        }

        // Perform a pointwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void pointwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread * M_thread; i++) {
                C_frag[i] = op(C_frag[i]);
            }
        }

        // Perform a rowwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void rowwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int row = i * N_warp + thread_y;
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    C_frag[i * M_thread + j] = op(C_frag[i * M_thread + j], row);
                }
            }
        }

        // Copy C from registers to shared memory
        __device__ void store(scalar_t* C_sm_ptr) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread ; j++) {
                    C_sm_ptr[(thread_y + i * N_warp) * M_tile + j * M_warp + thread_x]
                      = C_frag[i * M_thread + j];
                }
            }
        }

        __device__ void store_transpose(scalar_t* C_sm_ptr) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread ; j++) {
                    C_sm_ptr[thread_y + i * N_warp + (j * M_warp + thread_x) * N_tile]
                      = C_frag[i * M_thread + j];
                }
            }
        }
    };

    using namespace nvcuda;
    template<>
    struct warp_tile<c10::Half> {
        // How much data is processed by a single thread:
        static constexpr int N_thread = 2;
        static constexpr int M_thread = 1;

        // Thread layout within a warp:
        static constexpr int N_warp = 16;
        static constexpr int M_warp = 16;

        // Warp layout within a block:
        static constexpr int N_block = 2;
        static constexpr int M_block = 4;

        // Dimensions of the tile, in threads:
        static constexpr int N_tile = N_warp * N_block * N_thread;
        static constexpr int M_tile = M_warp * M_block * M_thread;
        static constexpr int K_tile = 16;

        // Registers:
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag[N_thread];

        int warp_x;   // x offset of the warp within the block tile
        int warp_y;   // y offset of the warp within the block tile

        __device__ warp_tile() {
            int warp_id = threadIdx.x / 32;
            warp_x = (warp_id % M_block);
            warp_y = (warp_id / M_block);
        }

        // Initialize C to all zeros
        __device__ void zero() {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                wmma::fill_fragment(C_frag[i], __float2half(0.f));
            }
        }

        // Performs C = A * B + C
        __device__ void mma(const c10::Half* A_sm_ptr, const c10::Half* B_sm_ptr, int k) {
            // Load a 1 x M fragment of B from shared memory to registers:
            wmma::load_matrix_sync(B_frag, reinterpret_cast<const half*>(B_sm_ptr) + warp_x * M_warp + k * M_tile, M_tile);

            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                // Load a N x 1 fragment of A from shared memory to registers:
                int y = (warp_y * N_thread + i) * N_warp;
                wmma::load_matrix_sync(A_frag, reinterpret_cast<const half*>(A_sm_ptr) + y + k * N_tile, N_tile);

                // Compute:
                wmma::mma_sync(C_frag[i], A_frag, B_frag, C_frag[i]);
            }
        }

        // Perform a pointwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void pointwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < C_frag[i].num_elements; j++) {
                    C_frag[i].x[j] = op(C_frag[i].x[j]);
                }
            }
        }

        __device__ int getWarpRow(int i) {
            int tid = threadIdx.x % 32;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
                return (tid & 3) + ((tid & 4) << 1) + ((tid & 16) >> 2); // half
                //return (tid & 16) / 4 + 2 * (tid & 4) + (tid & 1) + (i & 2); // float
            #else*/
                return (i & 2) * 4 + tid / 4;
            #endif
        }

        __device__ int getWarpCol(int i) {
            int tid = threadIdx.x % 32;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
                return (i & 7) + (tid & 8); // half
                //return (tid & 10) + (i & 5); // float
            #else
                return (tid % 4) * 2 + i % 2 + (i & 4) * 2;
            #endif
        }

        // Perform a rowwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void rowwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < C_frag[i].num_elements; j++) {
                    int row = getWarpRow(j) + i * 16 + warp_y * 32;
                    C_frag[i].x[j] = op(C_frag[i].x[j], row);
                }
            }
        }

        // Copy C from registers to shared memory
        __device__ void store(c10::Half* C_sm_ptr) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int y = (warp_y * N_thread + i) * N_warp;
                wmma::store_matrix_sync(reinterpret_cast<half*>(C_sm_ptr) + y * M_tile + warp_x * M_warp, C_frag[i], M_tile, wmma::mem_row_major);
            }
        }

        __device__ void store_transpose(c10::Half* C_sm_ptr) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int y = (warp_y * N_thread + i) * N_warp;
                wmma::store_matrix_sync(reinterpret_cast<half*>(C_sm_ptr) + y + warp_x * M_warp * N_tile, C_frag[i], N_tile, wmma::mem_col_major);
            }
        }
    };
} // namespace mma