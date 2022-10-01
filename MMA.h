#pragma once

#include<mma.h>

namespace mma {
    template<typename scalar_t, int N_tile_, int M_tile_>
    struct warp_tile {
        // Dimensions of the tile, in threads:
        static constexpr int N_tile = N_tile_;
        static constexpr int M_tile = M_tile_;
        static constexpr int K_tile = 1;

        // Warp layout within a block:
        static constexpr int N_block = 2;
        static constexpr int M_block = 4;

        // Thread layout within a warp:
        static constexpr int N_warp = 8;
        static constexpr int M_warp = 4;

        // How much data is processed by a single thread:
        static constexpr int N_thread = N_tile / (N_warp * N_block);
        static constexpr int M_thread = M_tile / (M_warp * M_block);

        static_assert(N_warp * N_block * N_thread == N_tile);
        static_assert(M_warp * M_block * M_thread == M_tile);
        static_assert(N_warp * M_warp == 32);
        static_assert(N_block * M_block * N_warp * M_warp == 256); // blockDim.x

        // Registers:
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
        template<typename fragA, typename fragB>
        __device__ void mma(fragA& A_sm, fragB& B_sm, int ka0, int kb0, int D) {
            float A_frag[N_thread]; // N x 1 fragment
            float B_frag[M_thread]; // 1 x M fragment
            
            for (int k = 0; k < D; k += K_tile) {
                // Load a N x 1 fragment of A from shared memory to registers:
                #pragma unroll
                for (int i = 0; i < N_thread; i++) {
                    A_frag[i] = A_sm(i * N_warp + thread_y, ka0 + k);
                }

                // Load a 1 x M fragment of B from shared memory to registers:
                #pragma unroll
                for (int i = 0; i < M_thread; i++) {
                    B_frag[i] = B_sm(i * M_warp + thread_x, kb0 + k);
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
        }

        // Perform a pointwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void pointwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int row = i * N_warp + thread_y;
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    int col = j * M_warp  + thread_x;
                    C_frag[i * M_thread + j] = op(C_frag[i * M_thread + j], col, row);
                }
            }
        }

        // Copy C from registers to shared memory
        template<typename shared_fragment>
        __device__ void store(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread ; j++) {
                    C_sm(j * M_warp + thread_x, i * N_warp + thread_y) = C_frag[i * M_thread + j];
                }
            }
        }

        template<typename shared_fragment>
        __device__ void store_transpose(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread ; j++) {
                    C_sm(i * N_warp + thread_y, j * M_warp + thread_x) = C_frag[i * M_thread + j];
                }
            }
        }

        // Stream C from registers to global memory using temporary shared memory buffer
        template<typename accessor, typename shared_fragment>
        __device__ void store(accessor gmem, shared_fragment& smem, int tile_x, int tile_y) {
            store(smem);
            __syncthreads();
            smem.store(gmem, tile_x, tile_y);
        }
    };

    using namespace nvcuda;
    template<int N_tile_, int M_tile_>
    struct warp_tile<c10::Half, N_tile_, M_tile_> {
        // Dimensions of the tile, in threads:
        static constexpr int N_tile = N_tile_;
        static constexpr int M_tile = M_tile_;
        static constexpr int K_tile = 16;

        // Warp layout within a block:
        static constexpr int N_block = 2;
        static constexpr int M_block = 4;

        // Thread layout within a warp:
        static constexpr int N_warp = 16;
        static constexpr int M_warp = 16;

        // How much data is processed by a single thread:
        static constexpr int N_thread = N_tile / (N_warp * N_block);
        static constexpr int M_thread = M_tile / (M_warp * M_block);

        static_assert(N_warp * N_block * N_thread == N_tile);
        static_assert(M_warp * M_block * M_thread == M_tile);

        // Registers:
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag[N_thread * M_thread];

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
        template<typename fragA, typename fragB>
        __device__ void mma(fragA& A_sm, fragB& B_sm, int ka0, int kb0, int D) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag;

            for (int k = 0; k < D; k += K_tile) {
                // Load a 1 x M fragment of B from shared memory to registers:
                wmma::load_matrix_sync(B_frag, reinterpret_cast<const half*>(&B_sm(warp_x * M_warp, kb0 + k)), B_sm.stride);

                #pragma unroll
                for (int i = 0; i < N_thread; i++) {
                    // Load a N x 1 fragment of A from shared memory to registers:
                    int y = (warp_y * N_thread + i) * N_warp;
                    wmma::load_matrix_sync(A_frag, reinterpret_cast<const half*>(&A_sm(y, ka0 + k)), A_sm.stride);

                    // Compute:
                    wmma::mma_sync(C_frag[i], A_frag, B_frag, C_frag[i]);
                }
            }
        }

        // Perform a pointwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void pointwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < C_frag[i].num_elements; j++) {
                    int col = getWarpCol(j) + warp_x * 16;
                    int row = getWarpRow(j) + i * 16 + warp_y * 32;
                    C_frag[i].x[j] = op(C_frag[i].x[j], col, row);
                }
            }
        }

        __device__ int getWarpRow(int i) {
            int tid = threadIdx.x % 32;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
                return (tid & 3) + ((tid & 4) << 1) + ((tid & 16) >> 2); // half
                //return (tid & 16) / 4 + 2 * (tid & 4) + (tid & 1) + (i & 2); // float
            #else
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

        // Copy C from registers to shared memory
        template<typename shared_fragment>
        __device__ void store(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int y = (warp_y * N_thread + i) * N_warp;
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    int x = (warp_x * M_thread + j) * M_warp;
                    wmma::store_matrix_sync(reinterpret_cast<half*>(&C_sm(x, y)), C_frag[i * M_thread + j], shared_fragment::stride, wmma::mem_row_major);
                }
            }
        }

        template<typename shared_fragment>
        __device__ void store_transpose(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int y = (warp_y * N_thread + i) * N_warp;
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    int x = (warp_x * M_thread + j) * M_warp;
                    wmma::store_matrix_sync(reinterpret_cast<half*>(&C_sm(y, x)), C_frag[i * M_thread + j], shared_fragment::stride, wmma::mem_col_major);
                }
            }
        }

        // Stream C from registers to global memory using temporary shared memory buffer
        template<typename accessor, typename shared_fragment>
        __device__ void store(accessor gmem, shared_fragment& smem, int tile_x, int tile_y) {
            store(smem);
            __syncthreads();
            smem.store(gmem, tile_x, tile_y);
        }
    };
} // namespace mma