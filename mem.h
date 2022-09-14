#pragma once

namespace mem {
    template<typename T>
    struct shared_fragment {
        T* smem;
        int N;
        int M;

        __device__ shared_fragment(char* shared_base, int N, int M)
          : smem(reinterpret_cast<T*>(shared_base)), N(N), M(M) { }

        __device__ void load(const T* gmem) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                smem[i] = gmem[i];
            }
        }

        template<typename accessor>
        __device__ void load(accessor gmem, int tile_x, int tile_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % M;
                int y = i / M;
                smem[i] = gmem[y+tile_y*N][x+tile_x*M];
            }
        }

        __device__ void load_transpose(const T* gmem) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int y = i % M;
                int x = i / M;
                smem[y * N + x] = gmem[i];
            }
        }

        template<typename accessor>
        __device__ void load_transpose(accessor gmem, int tile_x, int tile_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int y = i % M;
                int x = i / M;
                smem[y * N + x] = gmem[x+tile_y*N][y+tile_x*M];
            }
        }

        __device__ void store(T* gmem) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                gmem[i] = smem[i];
            }
        }

        template<typename accessor>
        __device__ void store(accessor gmem, int tile_x, int tile_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % M;
                int y = i / M;
                gmem[y+tile_y*N][x+tile_x*M] = smem[i];
            }
        }

        __device__ unsigned size() {
            return N * M;
        }

        __device__ char* next() {
            return reinterpret_cast<char*>(smem + size());
        }
    };
} // namespace mem