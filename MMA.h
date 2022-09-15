#pragma once

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
        __device__ void rowwise(int row, F&& op) {
            #pragma unroll
            for (int i = 0; i < M_thread; i++) {
                C_frag[row * M_thread + i] = op(C_frag[row * M_thread + i]);
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
} // namespace mma